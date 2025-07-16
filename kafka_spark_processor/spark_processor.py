# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, from_json, to_timestamp
# from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
# import os

# # Get configuration from environment variables
# kafka_host = os.environ.get("KAFKA_HOST", "kafka")
# kafka_port = os.environ.get("KAFKA_PORT", "9092")
# cassandra_host = os.environ.get("CASSANDRA_HOST", "cassandra")
# cassandra_port = os.environ.get("CASSANDRA_PORT", "9042")
# kafka_servers = f"{kafka_host}:{kafka_port}"

# print(f"Kafka connected successfully: {kafka_servers}")
# print(f"Cassandra connected successfully: {cassandra_host}:{cassandra_port}")

# # Create SparkSession
# spark = SparkSession.builder \
#     .appName("CoinbasePipeline") \
#     .config("spark.cassandra.connection.host", cassandra_host) \
#     .config("spark.cassandra.connection.port", cassandra_port) \
#     .getOrCreate()

# # Define the schema for exchange API
# schema = StructType([
#     StructField("type", StringType(), True),
#     StructField("sequence", LongType(), True),
#     StructField("product_id", StringType(), True),
#     StructField("price", StringType(), True),
#     StructField("open_24h", StringType(), True),
#     StructField("volume_24h", StringType(), True),
#     StructField("low_24h", StringType(), True),
#     StructField("high_24h", StringType(), True),
#     StructField("volume_30d", StringType(), True),
#     StructField("best_bid", StringType(), True),
#     StructField("best_ask", StringType(), True),
#     StructField("side", StringType(), True),
#     StructField("time", StringType(), True),
#     StructField("trade_id", LongType(), True),
#     StructField("last_size", StringType(), True)
# ])

# # Read stream from Kafka
# df = spark.readStream \
#     .format("kafka") \
#     .option("kafka.bootstrap.servers", kafka_servers) \
#     .option("subscribe", "coin-data") \
#     .option("startingOffsets", "earliest") \
#     .option("kafka.security.protocol", "PLAINTEXT") \
#     .load()

# # Parse JSON from Kafka
# json_df = df.select(
#     from_json(col("value").cast("string"), schema).alias("data")
# ).select("data.*")

# # Explode the events array
# processed_df = json_df.select(
#     "product_id",
#     col("price").cast("double").alias("price"),
#     to_timestamp(col("time")).alias("time")
# ).where(col("type") == "ticker")

# # Checkpoint path
# checkpoint_path = "/tmp/spark-checkpoint"

# # Save to Cassandra
# cassandra_query = processed_df.writeStream \
#     .foreachBatch(lambda batch_df, batch_id: 
#         batch_df.write \
#             .format("org.apache.spark.sql.cassandra") \
#             .option("keyspace", "coinbase") \
#             .option("table", "prices") \
#             .mode("append") \
#             .save()
#     ) \
#     .option("checkpointLocation", checkpoint_path) \
#     .start()

# print("Writing data to Cassandra...")

# # Wait for queries to complete
# spark.streams.awaitAnyTermination()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr, to_timestamp, when
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, ArrayType
import os

# Get configuration from environment variables
kafka_host = os.environ.get("KAFKA_HOST", "kafka")
kafka_port = os.environ.get("KAFKA_PORT", "9092")
cassandra_host = os.environ.get("CASSANDRA_HOST", "cassandra")
cassandra_port = os.environ.get("CASSANDRA_PORT", "9042")
kafka_servers = f"{kafka_host}:{kafka_port}"

print(f"Kafka connected successfully: {kafka_servers}")
print(f"Cassandra connected successfully: {cassandra_host}:{cassandra_port}")

# Create SparkSession
spark = SparkSession.builder \
    .appName("CoinbaseDataProcessor") \
    .config("spark.cassandra.connection.host", cassandra_host) \
    .config("spark.cassandra.connection.port", cassandra_port) \
    .getOrCreate()

# Define schema for ticker data from Advanced Trade API
ticker_schema = StructType([
    StructField("type", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("price", StringType(), True),
    StructField("volume_24h", StringType(), True),
    StructField("low_24h", StringType(), True),
    StructField("high_24h", StringType(), True),
    StructField("low_52w", StringType(), True),
    StructField("high_52w", StringType(), True),
    StructField("price_percent_chg_24h", StringType(), True),
    StructField("volume_percent_chg_24h", StringType(), True),
    StructField("price_change_24h", StringType(), True),
    StructField("volume_change_24h", StringType(), True),
    StructField("time", StringType(), True)
])

# Define schema for candle data
candle_schema = StructType([
    StructField("start", StringType(), True),
    StructField("high", StringType(), True),
    StructField("low", StringType(), True),
    StructField("open", StringType(), True),
    StructField("close", StringType(), True),
    StructField("volume", StringType(), True),
    StructField("product_id", StringType(), True)
])

# Function to process ticker data
def process_ticker_data():
    # Read ticker data from Kafka
    ticker_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", "coin-data") \
        .option("startingOffsets", "earliest") \
        .option("kafka.security.protocol", "PLAINTEXT") \
        .load()
    
    # Parse JSON
    parsed_ticker_df = ticker_df.select(
        from_json(col("value").cast("string"), ticker_schema).alias("ticker")
    ).select("ticker.*")
    
    # Process ticker data
    processed_ticker_df = parsed_ticker_df.select(
        col("product_id"),
        to_timestamp(col("time")).alias("time"),
        col("price").cast("double").alias("price")
    )
    
    # Write to Cassandra
    ticker_query = processed_ticker_df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: 
            batch_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .option("keyspace", "coinbase") \
                .option("table", "prices") \
                .mode("append") \
                .save()
        ) \
        .option("checkpointLocation", "/tmp/spark-ticker-checkpoint") \
        .start()
    
    return ticker_query

# Function to process candle data
def process_candle_data():
    # Read candle data from Kafka
    candle_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", "coin-data-model") \
        .option("startingOffsets", "earliest") \
        .option("kafka.security.protocol", "PLAINTEXT") \
        .load()
    
    # Parse JSON
    parsed_candle_df = candle_df.select(
        from_json(col("value").cast("string"), candle_schema).alias("candle")
    ).select("candle.*")
    
    # Process candle data
    processed_candle_df = parsed_candle_df.select(
        col("product_id"),
        # Convert start timestamp (Unix timestamp in seconds)
        when(col("start").cast("long").isNotNull(), 
             to_timestamp(col("start").cast("long"))).otherwise(
             to_timestamp(col("start"))).alias("start_time"),  
        col("open").cast("double").alias("open"),
        col("high").cast("double").alias("high"),
        col("low").cast("double").alias("low"),
        col("close").cast("double").alias("close"),
        col("volume").cast("double").alias("volume")
    )
    
    # Write to Cassandra
    candle_query = processed_candle_df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: 
            batch_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .option("keyspace", "coinbase") \
                .option("table", "candles") \
                .mode("append") \
                .save()
        ) \
        .option("checkpointLocation", "/tmp/spark-candles-checkpoint") \
        .start()
    
    return candle_query

# Start both processors
print("Starting ticker data processor...")
ticker_query = process_ticker_data()

print("Starting candle data processor...")
candle_query = process_candle_data()

print("Both processors running. Waiting for termination...")
spark.streams.awaitAnyTermination()