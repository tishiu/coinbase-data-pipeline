# cassandra/Dockerfile
FROM cassandra:latest

# Copy initialization scripts
COPY /scripts/ /docker-entrypoint-initdb.d/

# Set permissions
RUN chmod +x /docker-entrypoint-initdb.d/*.sh

# Set environment variables
ENV CASSANDRA_CLUSTER_NAME=coinbase_cluster
ENV CASSANDRA_KEYSPACE=coinbase

# Expose Cassandra ports
EXPOSE 9042 

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD nodetool status | grep -q '^UN' || exit 1