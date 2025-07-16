import os
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import glob
from lstm_attention_model import LSTMAttentionModel
from rwkv_ts_model import CryptoRWKV_TS
from lstm_model import LSTMModel
from cnn_lstm_model import CNNLSTMModel
from cnn_lstm_attention_model import CNNLSTMAttentionModel
from lstm_attention_hybrid_model import LSTMAttentionHybrid
from data_loader import CryptoDataLoader
from optimize_model import OptimizedLSTMAttentionModel
from utils import TrainingTracker, EarlyStopper, CompositeLoss, QuantileLoss, DirectionLoss
import torch.nn.functional as F
from typing import Dict, Any
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
import math
from sklearn.model_selection import TimeSeriesSplit

# Cấu hình logging và suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveHuberLoss(nn.Module):
    def __init__(self, initial_delta=1.0):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(initial_delta))
        self.delta.requires_grad = False

    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        condition = residual < self.delta
        loss = torch.where(
            condition,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        return loss.mean()
    
    def update_delta(self, new_delta):
        self.delta.fill_(new_delta)

class TrainConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.epochs = config_dict['training']['epochs']
        self.batch_size = config_dict['training']['batch_size']
        self.lr = config_dict['training']['lr']
        self.device = torch.device(config_dict['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.data_path = config_dict['data']['path']
        self.log_dir = config_dict['training'].get('log_dir', 'logs')
        self.checkpoint_dir = config_dict['training'].get('checkpoint_dir', 'checkpoints')
        self.resume = config_dict['training'].get('resume', None)
        self.checkpoint_interval = config_dict['training'].get('checkpoint_interval', 5)
        self.loss_fn = config_dict['training'].get('loss_fn', 'huber')
        self.huber_delta = config_dict['training'].get('huber_delta', 0.5)
        self.use_amp = config_dict['training'].get('use_amp', False)
        self.grad_accum_steps = config_dict['training'].get('grad_accum_steps', 4)
        self.ema_decay = config_dict['training'].get('ema_decay', 0.999)
        self.use_swa = config_dict['training'].get('use_swa', False)
        self.swa_lr = config_dict['training'].get('swa_lr', 0.05)
        self.swa_start_ratio = config_dict['training'].get('swa_start_ratio', 0.75)
        self.warmup_epochs = config_dict['training'].get('warmup_epochs', 5)

def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5,
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class SWAWrapper:
    def __init__(self, model, optimizer, config: TrainConfig):
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            optimizer, 
            swa_lr=config.swa_lr,
            anneal_epochs=5
        )
        self.swa_start = int(config.epochs * config.swa_start_ratio)
        self.swa_activated = False

    def update(self, model, epoch):
        if epoch >= self.swa_start and not self.swa_activated:
            logger.info(f"Starting SWA at epoch {epoch+1}")
            self.swa_activated = True
        
        if self.swa_activated:
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
            return True
        return False
    
    def finalize(self, data_loader):
        if self.swa_activated:
            logger.info("Finalizing SWA with BN update...")
            torch.optim.swa_utils.update_bn(data_loader, self.swa_model)
            return self.swa_model.module
        return None

def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    preds = []
    targets = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating', leave=False):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += loss.item()
                
                preds.append(pred.cpu().numpy().reshape(-1))
                targets.append(y.cpu().numpy().reshape(-1))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    metrics = {
        'loss': total_loss / len(data_loader),
        'mse': mean_squared_error(targets, preds),
        'mae': mean_absolute_error(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'smape': 100 * np.mean(2.0 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + 1e-8)),
        'r2': r2_score(targets, preds)
    }
    
    if len(preds) > 1:
        pred_directions = np.sign(preds[1:] - preds[:-1])
        true_directions = np.sign(targets[1:] - targets[:-1])
        metrics['direction_acc'] = np.mean(pred_directions == true_directions)
    
    print(f"[Eval] Loss: {metrics['loss']:.4f} | MSE: {metrics['mse']:.4f} | "
          f"MAE: {metrics['mae']:.4f} | SMAPE: {metrics['smape']:.2f}% | "
          f"R2: {metrics['r2']:.4f} | Direction Acc: {metrics.get('direction_acc', 0):.2%}")
    
    return metrics['loss']

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*/epoch_*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'val_loss': checkpoint.get('val_loss', float('inf'))
    }

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")
    
def train(config_path: str = 'configs/train_config.yaml'):
    try:
        # 1. Load config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = TrainConfig(config_dict)

        # 2. Khởi tạo hệ thống
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracker = TrainingTracker(config_dict)
        stopper = EarlyStopper(config_dict)
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        # 3. Chuẩn bị dữ liệu
        logger.info("Loading data...")
        data_loader = CryptoDataLoader(config_path=config_path)
        train_loader = data_loader.train_loader
        val_loader = data_loader.test_loader

        # 4. Khởi tạo model
        model_type = config_dict['model'].get('model_type', 'lstm').lower()
        if model_type == 'lstm_attention':
            model = LSTMAttentionModel(config_dict).to(config.device)
        elif model_type == 'optimize':
            model = OptimizedLSTMAttentionModel(config_dict).to(config.device)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMModel(config_dict).to(config.device)
        elif model_type == 'cnn_lstm_attention':
            model = CNNLSTMAttentionModel(config_dict).to(config.device)
        elif model_type == 'lstm_hybridattention':
            model = LSTMAttentionHybrid(config_dict).to(config.device)
        else:
            model = LSTMModel(config_dict).to(config.device)
        logger.info(f"Using model: {model_type}")

        # 5. Khởi tạo loss function
        if config.loss_fn.lower() == "huber":
            loss_fn = AdaptiveHuberLoss(initial_delta=config.huber_delta)
            logger.info(f"Using HuberLoss with initial delta={config.huber_delta}")
            
            with torch.no_grad():
                sample = next(iter(train_loader))
                pred = model(sample['x'].to(config.device))
                errors = torch.abs(pred - sample['y'].to(config.device))
                delta = torch.quantile(errors, 0.8).item()
                loss_fn.update_delta(delta)
                logger.info(f"Auto-adjusted delta to: {delta:.4f}")
        else:
            loss_fn = CompositeLoss(
                losses=[nn.MSELoss(), DirectionLoss(alpha= 0.3)],
                weights=[1.0, 0.0]
            )
            logger.info("Using CompositeLoss (MSE + Quantile)")

        # 6. Tối ưu hóa
        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-3
        )
        
        # 7. Thiết lập scheduler
        total_steps = len(train_loader) * config.epochs
        warmup_steps = len(train_loader) * config.warmup_epochs
        
        if config.use_swa:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5
            )
            swa = SWAWrapper(model, optimizer, config)
        else:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config.lr*10,
                steps_per_epoch=len(train_loader),
                epochs=config.epochs,
                pct_start=0.3
            )
            swa = None

        # 8. Resume training nếu có
        start_epoch = 0
        best_loss = float('inf')
        checkpoint_path = None
        if config.resume == 'auto':
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        elif config.resume:
            checkpoint_path = config.resume
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"Resumed training from epoch {start_epoch}")

        # 9. Vòng lặp training
        train_losses = []
        val_losses = []
        
        for epoch in range(start_epoch, config.epochs):
            train_loader.dataset.set_epoch(epoch, config.epochs)
            model.train()
            epoch_loss = 0
            optimizer.zero_grad()
            
            # Training phase với gradient accumulation
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{config.epochs}") as tepoch:
                for i, batch in enumerate(tepoch):
                    x = batch['x'].to(config.device)
                    y = batch['y'].to(config.device)
                    
                    # Mixed precision forward
                    with torch.cuda.amp.autocast(enabled=config.use_amp):
                        pred = model(x)
                        loss = loss_fn(pred, y) / config.grad_accum_steps
                    
                    # Backward với gradient scaling
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm= 1.0
                        )
                        
                        # Cập nhật weights
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    epoch_loss += loss.item() * config.grad_accum_steps
                    tepoch.set_postfix(loss=loss.item() * config.grad_accum_steps)

            # Validation phase
            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, config.device, loss_fn)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            # Cập nhật SWA nếu được kích hoạt
            if config.use_swa:
                swa.update(model, epoch)

            # Cập nhật delta cho Huber Loss
            if isinstance(loss_fn, AdaptiveHuberLoss):
                with torch.no_grad():
                    preds = model(torch.cat([b['x'] for b in train_loader], dim=0).to(config.device))
                    targets = torch.cat([b['y'] for b in train_loader], dim=0).to(config.device)
                    errors = torch.abs(preds - targets)
                    new_delta = torch.quantile(errors, 0.8).item()
                    loss_fn.update_delta(new_delta)
                    tracker.log("Loss/delta", new_delta, epoch)

            # Logging
            tracker.log("Loss/train", avg_train_loss, epoch)
            tracker.log("Loss/val", val_loss, epoch)
            tracker.log("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)

            logger.info(f"Epoch {epoch+1}/{config.epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Lưu checkpoint
            if epoch % config.checkpoint_interval == 0 or val_loss < best_loss:
                if val_loss < best_loss:
                    best_loss = val_loss
                    prefix = "best_"
                else:
                    prefix = ""
                
                checkpoint_path = f"{config.checkpoint_dir}/{timestamp}/{prefix}epoch_{epoch}.pt"
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss,
                    'config': config_dict
                }, checkpoint_path)

            # Early stopping
            if stopper.check(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Finalize SWA
        if config.use_swa:
            final_model = swa.finalize(train_loader)
            if final_model is not None:
                final_val_loss = evaluate(final_model, val_loader, config.device, loss_fn)
                logger.info(f"Final SWA Val Loss: {final_val_loss:.4f}")
                torch.save(final_model.state_dict(), f"final_swa_model_{timestamp}.pt")
        else:
            torch.save(model.state_dict(), f"final_model_{timestamp}.pt")

        # Visualize kết quả
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curve.png')
        plt.show()

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        tracker.close()
        logger.info("Training completed")

if __name__ == "__main__":
    train()
    
# using ema model    
# def train(config_path: str = 'configs/train_config.yaml'):
#     try:
#         # 1. Load config
#         with open(config_path) as f:
#             config_dict = yaml.safe_load(f)
#         config = TrainConfig(config_dict)

#         # 2. Khởi tạo hệ thống
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         tracker = TrainingTracker(config_dict)
#         stopper = EarlyStopper(config_dict)
#         scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

#         # 3. Chuẩn bị dữ liệu
#         logger.info("Loading data...")
#         data_loader = CryptoDataLoader(config_path=config_path)
#         train_loader = data_loader.train_loader
#         val_loader = data_loader.test_loader

#         # 4. Khởi tạo model
#         model_type = config_dict['model'].get('model_type', 'lstm').lower()
#         if model_type == 'lstm_attention':
#             model = LSTMAttentionModel(config_dict).to(config.device)
#         elif model_type == 'optimize':
#             model = OptimizedLSTMCNNAttention(config_dict).to(config.device)
#         elif model_type == 'cnn_lstm':
#             model = CNNLSTMModel(config_dict).to(config.device)
#         elif model_type == 'cnn_lstm_attention':
#             model = LSTMCNNAttentionModel(config_dict).to(config.device)
#         elif model_type == 'lstm_hybridattention':
#             model = LSTMAttentionHybrid(config_dict).to(config.device)
#         else:
#             model = LSTMModel(config_dict).to(config.device)
#         logger.info(f"Using model: {model_type}")

#         # 5. Khởi tạo loss function
#         if config.loss_fn.lower() == "huber":
#             loss_fn = AdaptiveHuberLoss(initial_delta=config.huber_delta)
#             logger.info(f"Using HuberLoss with initial delta={config.huber_delta}")
            
#             with torch.no_grad():
#                 sample = next(iter(train_loader))
#                 pred = model(sample['x'].to(config.device))
#                 errors = torch.abs(pred - sample['y'].to(config.device))
#                 delta = torch.quantile(errors, 0.8).item()
#                 loss_fn.update_delta(delta)
#                 logger.info(f"Auto-adjusted delta to: {delta:.4f}")
#         else:
#             loss_fn = CompositeLoss(
#                 losses=[nn.MSELoss(), QuantileLoss(quantiles=[0.1, 0.5, 0.9])],
#                 weights=[0.7, 0.3]
#             )
#             logger.info("Using CompositeLoss (MSE + Quantile)")

#         # 6. Tối ưu hóa
#         optimizer = AdamW(
#             model.parameters(),
#             lr=config.lr,
#             betas=(0.9, 0.999),
#             weight_decay=1e-4
#         )
        
#         scheduler = OneCycleLR(
#             optimizer,
#             max_lr=config.lr*10,
#             steps_per_epoch=len(train_loader),
#             epochs=config.epochs,
#             pct_start=0.3,
#             div_factor=25,
#             final_div_factor=100
#         )

#         # 7. Khởi tạo EMA
#         ema = ModelEMA(model, decay=config.ema_decay)

#         # 8. Resume training nếu có
#         start_epoch = 0
#         best_loss = float('inf')
#         checkpoint_path = None
#         if config.resume == 'auto':
#             checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
#         elif config.resume:
#             checkpoint_path = config.resume
        
#         if checkpoint_path:
#             checkpoint = torch.load(checkpoint_path, map_location=config.device)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             if 'scheduler_state_dict' in checkpoint:
#                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#             start_epoch = checkpoint['epoch'] + 1
#             best_loss = checkpoint.get('best_loss', float('inf'))
#             logger.info(f"Resumed training from epoch {start_epoch}")

#         # 9. Vòng lặp training
#         train_losses = []
#         val_losses = []
        
#         for epoch in range(start_epoch, config.epochs):
#             train_loader.dataset.set_epoch(epoch, config.epochs)
#             model.train()
#             epoch_loss = 0
#             optimizer.zero_grad()
            
#             # 9.1 Training phase với gradient accumulation
#             with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{config.epochs}") as tepoch:
#                 for i, batch in enumerate(tepoch):
#                     x = batch['x'].to(config.device)
#                     y = batch['y'].to(config.device)
                    
#                     # Mixed precision forward
#                     with torch.cuda.amp.autocast(enabled=config.use_amp):
#                         pred = model(x)
                        
#                         # Ensure target matches prediction length
#                         if y.shape[1] > pred.shape[1]:
#                             y = y[:, :pred.shape[1], :]  # Truncate target if longer
#                         elif y.shape[1] < pred.shape[1]:
#                             # Pad target if shorter (though this shouldn't happen if data is prepared correctly)
#                             pad_size = pred.shape[1] - y.shape[1]
#                             y = F.pad(y, (0, 0, 0, pad_size), "constant", 0)
                        
#                         loss = loss_fn(pred, y) / config.grad_accum_steps
                    
#                     # Backward với gradient scaling
#                     scaler.scale(loss).backward()
                    
#                     if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(train_loader):
#                         # Gradient clipping
#                         scaler.unscale_(optimizer)
#                         torch.nn.utils.clip_grad_norm_(
#                             model.parameters(),
#                             max_norm=0.5 * (1 + epoch/config.epochs)
#                         )
                        
#                         # Cập nhật weights
#                         scaler.step(optimizer)
#                         scaler.update()
#                         optimizer.zero_grad()
#                         scheduler.step()
                        
#                         # Cập nhật EMA
#                         ema.update(model)
                    
#                     epoch_loss += loss.item() * config.grad_accum_steps
#                     tepoch.set_postfix(loss=loss.item() * config.grad_accum_steps)

#             # 9.2 Evaluation phase
#             avg_train_loss = epoch_loss / len(train_loader)
            
#             # Validate với cả model và EMA model
#             val_loss = evaluate(model, val_loader, config.device, loss_fn)
#             ema_val_loss = evaluate(ema.module, val_loader, config.device, loss_fn)
            
#             # Chọn model tốt hơn
#             if ema_val_loss < val_loss:
#                 model.load_state_dict(ema.module.state_dict())
#                 val_loss = ema_val_loss
#                 logger.info("Using EMA model for better validation loss")
            
#             train_losses.append(avg_train_loss)
#             val_losses.append(val_loss)

#             # 9.3 Cập nhật delta cho Huber Loss
#             if isinstance(loss_fn, AdaptiveHuberLoss):
#                 with torch.no_grad():
#                     preds = model(torch.cat([b['x'] for b in train_loader], dim=0).to(config.device))
#                     targets = torch.cat([b['y'] for b in train_loader], dim=0).to(config.device)
#                     errors = torch.abs(preds - targets)
#                     new_delta = torch.quantile(errors, 0.8).item()
#                     loss_fn.update_delta(new_delta)
#                     tracker.log("Loss/delta", new_delta, epoch)

#             # 9.4 Logging
#             tracker.log("Loss/train", avg_train_loss, epoch)
#             tracker.log("Loss/val", val_loss, epoch)
#             tracker.log("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)

#             logger.info(f"Epoch {epoch+1}/{config.epochs} | "
#                       f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
#                       f"LR: {optimizer.param_groups[0]['lr']:.2e}")

#             # 9.5 Lưu checkpoint
#             if epoch % config.checkpoint_interval == 0 or val_loss < best_loss:
#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     prefix = "best_"
#                 else:
#                     prefix = ""
                
#                 checkpoint_path = f"{config.checkpoint_dir}/{timestamp}/{prefix}epoch_{epoch}.pt"
#                 save_checkpoint({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict(),
#                     'loss': avg_train_loss,
#                     'val_loss': val_loss,
#                     'best_loss': best_loss,
#                     'config': config_dict
#                 }, checkpoint_path)

#             # 9.6 Early stopping
#             if stopper.check(val_loss):
#                 logger.info(f"Early stopping triggered at epoch {epoch+1}")
#                 break

#         # 10. Visualize kết quả
#         plt.figure(figsize=(10, 6))
#         plt.plot(train_losses, label='Training Loss', color='blue')
#         plt.plot(val_losses, label='Validation Loss', color='red')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.title('Learning Curve')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig('learning_curve.png')
#         plt.show()

#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}", exc_info=True)
#         raise
#     finally:
#         tracker.close()
#         logger.info("Training completed")
