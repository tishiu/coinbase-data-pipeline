package main

import (
    "bytes"
    "fmt"
    "log"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/credentials"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
    "github.com/confluentinc/confluent-kafka-go/kafka"
)

func main() {
    // Kafka configuration from environment variables
    bootstrapServers := getEnv("BOOTSTRAP_SERVERS", "localhost:9092")
    
    // Create a single consumer for both topics
    c, err := kafka.NewConsumer(&kafka.ConfigMap{
        "bootstrap.servers": bootstrapServers,
        "group.id":          "s3-consumer-group",
        "auto.offset.reset": "earliest",
    })
    if err != nil {
        log.Fatalf("Failed to create consumer: %s", err)
    }
    defer c.Close()

    fmt.Printf("Created Kafka consumer with bootstrap servers: %s\n", bootstrapServers)
    
    // Subscribe to both topics
    c.SubscribeTopics([]string{"coin-data", "coin-data-model"}, nil)
    
    // S3/MinIO configuration from environment variables
    region := getEnv("AWS_REGION", "ap-southeast-1")
    endpoint := getEnv("AWS_ENDPOINT", "")
    accessKey := getEnv("AWS_ACCESS_KEY_ID", "minioadmin")
    secretKey := getEnv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    forcePathStyle := getEnv("AWS_S3_FORCE_PATH_STYLE", "true") == "true"
    bucketName := getEnv("S3_BUCKET", "ie212-coinbase-data")
    
    // Create AWS session for S3/MinIO
    awsConfig := &aws.Config{
        Region:           aws.String(region),
        Credentials:      credentials.NewStaticCredentials(accessKey, secretKey, ""),
        S3ForcePathStyle: aws.Bool(forcePathStyle),
    }
    
    // If endpoint is set, use it (for MinIO)
    if endpoint != "" {
        awsConfig.Endpoint = aws.String(endpoint)
    }
    
    sess, err := session.NewSession(awsConfig)
    if err != nil {
        log.Fatalf("Failed to create AWS session: %s", err)
    }
    
    s3Client := s3.New(sess)
    fmt.Printf("Connected successfully to S3/MinIO, bucket: %s\n", bucketName)
    
    // Set up signal handling for graceful shutdown
    sigchan := make(chan os.Signal, 1)
    signal.Notify(sigchan, syscall.SIGINT, syscall.SIGTERM)
    
    // Set timeout for polling
    timeout := 30 * time.Second
    
    // Poll messages from Kafka and write to S3/MinIO
    fmt.Println("Getting data from Kafka and writing to S3/MinIO...")
    run := true
    for run {
        select {
        case sig := <-sigchan:
            fmt.Printf("Caught signal %v: terminating\n", sig)
            run = false
        default:
            msg, err := c.ReadMessage(timeout)
            if err != nil {
                // Check timeout
                if err.(kafka.Error).Code() == kafka.ErrTimedOut {
                    fmt.Println("Timed out, continuing...")
                    continue
                }
                log.Printf("Consumer error: %v (%v)\n", err, msg)
                continue
            }
            
            // Determine the data type based on topic
            var dataType string
            if *msg.TopicPartition.Topic == "coin-data" {
                dataType = "ticker"
            } else if *msg.TopicPartition.Topic == "coin-data-model" {
                dataType = "candles"
            } else {
                dataType = "unknown"
            }
            
            // Create a unique S3 object key based on data type, product_id, and timestamp
            objectKey := fmt.Sprintf("%s/%s/%d.json", dataType, string(msg.Key), time.Now().UnixNano())
            
            // Write message to S3/MinIO
            _, err = s3Client.PutObject(&s3.PutObjectInput{
                Bucket:      aws.String(bucketName),
                Key:         aws.String(objectKey),
                Body:        bytes.NewReader(msg.Value),
                ContentType: aws.String("application/json"),
            })
            
            if err != nil {
                log.Printf("Failed to write to S3/MinIO: %s", err)
                continue
            }
            
            fmt.Printf("Saved %s data for %s to S3/MinIO: %s\n", dataType, string(msg.Key), objectKey)
        }
    }
    
    fmt.Println("Consumer stopped gracefully")
}

// Helper function to get environment variables with defaults
func getEnv(key, defaultValue string) string {
    value := os.Getenv(key)
    if value == "" {
        return defaultValue
    }
    return value
}