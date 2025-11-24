# Use official vLLM Docker image with CUDA support
FROM vllm/vllm-openai:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the vLLM startup script
COPY launch_server.sh .

# Make startup script executable
RUN chmod +x launch_server.sh

# Create model directory
RUN mkdir -p /app/models

# Expose the service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/v1/models || exit 1

# Start vLLM's built-in OpenAI API server
CMD ["./launch_server.sh"]
