# FROM grafana/grafana:latest
# # Install required plugin for cassandra
# RUN grafana-cli plugins install hadesarchitect-cassandra-datasource

# COPY config/datasource.yaml /etc/grafana/provisioning/datasources/
# COPY dashboards/dashboard.yaml /etc/grafana/provisioning/dashboards/
# COPY dashboards/dashboard.json /etc/grafana/provisioning/dashboards/
# COPY grafana.ini /etc/grafana/grafana.ini

# # Login credentials
# ENV GF_SECURITY_ADMIN_USER=admin
# ENV GF_SECURITY_ADMIN_PASSWORD=admin
# ENV GF_AUTH_ANONYMOUS_ENABLED=true

# USER grafana

# grafana/Dockerfile
FROM grafana/grafana:latest

# Install required plugin for cassandra
RUN grafana-cli plugins install hadesarchitect-cassandra-datasource

# Copy configuration files
COPY config/datasource.yaml /etc/grafana/provisioning/datasources/
COPY dashboards/dashboard.yaml /etc/grafana/provisioning/dashboards/

# Copy dashboard JSON files
COPY dashboards/dashboard.json /etc/grafana/provisioning/dashboards/
COPY dashboards/predict_dashboard.json /etc/grafana/provisioning/dashboards/

# Copy Grafana configuration
COPY grafana.ini /etc/grafana/grafana.ini

# Login credentials
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=admin
ENV GF_AUTH_ANONYMOUS_ENABLED=true

USER grafana