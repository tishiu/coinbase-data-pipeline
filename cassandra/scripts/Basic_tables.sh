#!/bin/bash
set -e

# Create keyspace if not exists
cqlsh -e "CREATE KEYSPACE IF NOT EXISTS $CASSANDRA_KEYSPACE WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};"

echo "Keyspace $CASSANDRA_KEYSPACE created successfully!"

# Create prices table
cqlsh -e "
CREATE TABLE IF NOT EXISTS $CASSANDRA_KEYSPACE.prices (
    product_id TEXT,
    time TIMESTAMP,
    price DOUBLE,
    PRIMARY KEY (product_id, time)
) WITH CLUSTERING ORDER BY (time DESC);"

# Create candles table:
cqlsh -e "
CREATE TABLE IF NOT EXISTS $CASSANDRA_KEYSPACE.candles (
    product_id TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    candle_size TEXT,
    channel TEXT,
    PRIMARY KEY (product_id, start_time)
) WITH CLUSTERING ORDER BY (start_time DESC);"

echo "Basic tables created successfully!"