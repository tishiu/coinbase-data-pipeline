#!/bin/bash

# Default values
CASSANDRA_HOST=${CASSANDRA_HOST:-cassandra}
CASSANDRA_PORT=${CASSANDRA_PORT:-9042}
CASSANDRA_KEYSPACE=${CASSANDRA_KEYSPACE:-coinbase}

echo "=== Setting up prediction tables in Cassandra ==="
echo "Host: $CASSANDRA_HOST"
echo "Port: $CASSANDRA_PORT"
echo "Keyspace: $CASSANDRA_KEYSPACE"

# Wait for Cassandra to be ready
echo "Waiting for Cassandra to be ready..."
until cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "DESCRIBE KEYSPACES" > /dev/null 2>&1; do
    echo "Waiting for Cassandra..."
    sleep 5
done

echo "Cassandra is ready!"

# Check if keyspace exists, create if it doesn't
KEYSPACE_EXISTS=$(cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "DESCRIBE KEYSPACES" | grep -c "$CASSANDRA_KEYSPACE")
if [ "$KEYSPACE_EXISTS" -eq 0 ]; then
    echo "Creating keyspace $CASSANDRA_KEYSPACE..."
    cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "CREATE KEYSPACE $CASSANDRA_KEYSPACE WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};"
else
    echo "Keyspace $CASSANDRA_KEYSPACE already exists."
fi

# Create predictions table
echo "Creating predictions table..."
cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "
CREATE TABLE IF NOT EXISTS $CASSANDRA_KEYSPACE.predictions (
    product_id TEXT,
    model_name TEXT,
    prediction_time TIMESTAMP,
    target_time TIMESTAMP,
    predicted_price DOUBLE,
    confidence_lower DOUBLE,
    confidence_upper DOUBLE,
    model_version TEXT,
    metadata TEXT,
    PRIMARY KEY ((product_id, model_name), prediction_time, target_time)
) WITH CLUSTERING ORDER BY (prediction_time DESC, target_time ASC);
"

# Create predictions_by_horizon table for easy querying
echo "Creating predictions_by_horizon table..."
cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "
CREATE TABLE IF NOT EXISTS $CASSANDRA_KEYSPACE.predictions_by_horizon (
    product_id TEXT,
    model_name TEXT,
    prediction_horizon INT,
    prediction_time TIMESTAMP,
    target_time TIMESTAMP,
    predicted_price DOUBLE,
    confidence_lower DOUBLE,
    confidence_upper DOUBLE,
    PRIMARY KEY ((product_id, model_name, prediction_horizon), prediction_time)
) WITH CLUSTERING ORDER BY (prediction_time DESC);
"

# Create model metrics table
echo "Creating model_metrics table..."
cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "
CREATE TABLE IF NOT EXISTS $CASSANDRA_KEYSPACE.model_metrics (
    product_id TEXT,
    model_name TEXT,
    evaluation_time TIMESTAMP,
    horizon INT,
    mae DOUBLE,
    rmse DOUBLE,
    mape DOUBLE,
    directional_accuracy DOUBLE,
    sample_count INT,
    PRIMARY KEY ((product_id, model_name), evaluation_time, horizon)
) WITH CLUSTERING ORDER BY (evaluation_time DESC, horizon ASC);
"

echo "Tables created successfully!"

# Verify tables
echo "Verifying tables..."
cqlsh $CASSANDRA_HOST $CASSANDRA_PORT -e "DESCRIBE TABLES FROM $CASSANDRA_KEYSPACE;"

echo "Setup prediction tables complete"