#!/bin/bash
set -e

# Wait for ArangoDB to be ready
echo "Waiting for ArangoDB to start..."
until curl --silent --fail http://localhost:8529/_api/version > /dev/null; do
  echo "ArangoDB is unavailable - sleeping"
  sleep 1
done

echo "ArangoDB is up - executing initialization script"

# Run the database creation script
arangosh \
  --server.endpoint tcp://127.0.0.1:8529 \
  --server.authentication false \
  --javascript.execute /docker-entrypoint-initdb.d/create-database.js

echo "Initialization completed" 