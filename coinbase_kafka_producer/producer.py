import os
import json
import websocket
import threading
import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

TOPIC = "coin-data"
CANDLES_TOPIC = "coin-data-model" 
BOOTSTRAP_SERVERS = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9092")
print(f"Attempting to connect to Kafka at: {BOOTSTRAP_SERVERS}")
PRODUCT_IDS = ["ETH-USD", "BTC-USD", "XRP-USD"]

# Updated to use Advanced Trade WebSocket API
COINBASE_ADVANCED_WS_URL = "wss://advanced-trade-ws.coinbase.com"

def create_producer():
    try:
        # Removed key_serializer to avoid None key encoding issues
        return KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=str.encode
        )
    except KafkaError as e:
        print(f"Failed to create Kafka producer: {e}")
        return None

def on_open(ws):
    print("WebSocket connection opened")
    
    # Subscribe to ticker channel
    ticker_subscribe_message = {
        "type": "subscribe",
        "product_ids": PRODUCT_IDS,
        "channel": "ticker"
    }
    
    # Subscribe to candles channel
    candles_subscribe_message = {
        "type": "subscribe",
        "product_ids": PRODUCT_IDS,
        "channel": "candles"
    }
    
    try:
        # Send ticker subscription
        ws.send(json.dumps(ticker_subscribe_message))
        print(f"Sent ticker subscription request for {PRODUCT_IDS}")
        
        # Send candles subscription
        ws.send(json.dumps(candles_subscribe_message))
        print(f"Sent candles subscription request for {PRODUCT_IDS}")
    except Exception as e:
        print(f"Error sending subscription: {e}")

def on_message(ws, message, producer, subscribed):
    # Check if message is valid
    if message is None or not isinstance(message, str) or not message.strip():
        print(f"Invalid message received: {repr(message)}")
        return

    try:
        data = json.loads(message)
        channel = data.get("channel")
        timestamp = data.get("timestamp")  # Get timestamp from message if available
        
        # Handle subscription confirmations
        if data.get("type") == "subscriptions":
            print(f"Subscription successful: {message}")
            subscribed[0] = True
            return
            
        # Handle subscription errors
        if data.get("type") == "error":
            print(f"Subscription failed: {message}")
            if producer:
                producer.close()
            ws.close()
            return
            
        # Process ticker messages
        if channel == "ticker" and producer is not None and "events" in data:
            for event in data.get("events", []):
                if "tickers" in event:
                    for ticker in event.get("tickers", []):
                        # Add timestamp from the event data, or current time if not available
                        if not "time" in ticker:
                            # Use timestamp from main message, or generate current time
                            current_time = datetime.datetime.utcnow().isoformat() + "Z"
                            ticker["time"] = timestamp if timestamp else current_time
                        
                        product_id = ticker.get("product_id", "unknown")
                        ticker_json = json.dumps(ticker)
                        print(f"Sending {product_id} ticker to Kafka: {ticker_json[:100]}...")
                        producer.send(TOPIC, key=product_id.encode(), value=ticker_json)
        
        # Process candles messages
        elif channel == "candles" and producer is not None and "events" in data:
            for event in data.get("events", []):
                if "candles" in event:
                    for candle in event.get("candles", []):
                        product_id = candle.get("product_id", "unknown")
                        candle_json = json.dumps(candle)
                        print(f"Sending {product_id} candle to Kafka: {candle_json[:100]}...")
                        producer.send(CANDLES_TOPIC, key=product_id.encode(), value=candle_json)
        else:
            print(f"Received message from channel {channel}: {message[:100]}...")
    except json.JSONDecodeError:
        print(f"Failed to parse message as JSON: {message[:100]}...")
    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed: {close_status_code}, {close_msg}")

def main():
    producer = create_producer()
    if producer is None:
        print("Cannot proceed without Kafka producer")
        return
        
    subscribed = [False]
    ws = websocket.WebSocketApp(
        COINBASE_ADVANCED_WS_URL,
        on_open=on_open,
        on_message=lambda ws, msg: on_message(ws, msg, producer, subscribed),
        on_error=on_error,
        on_close=on_close
    )
    
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    try:
        print("Producer running. Press Ctrl+C to stop...")
        ws_thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        if producer:
            producer.close()
        ws.close()

if __name__ == "__main__":
    main()