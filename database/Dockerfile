FROM postgis/postgis:15-3.4

# Install pgRouting extension
RUN apt-get update && \
    apt-get install -y postgresql-15-pgrouting && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*




