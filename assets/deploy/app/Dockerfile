# Use the official Node.js image from the Docker Hub
FROM node:18-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV NPM_CONFIG_YES=true
ENV PNPM_HOME=/pnpm
ENV PATH="$PNPM_HOME:$PATH"

# Set the working directory
WORKDIR /app

# Install pnpm globally with --force and yes flags
RUN npm install -g pnpm --force --yes

# Copy package files ONLY (for better Docker layer caching)
# Copy package.json (required) and pnpm-lock.yaml (optional)
COPY ./frontend/package.json ./
COPY ./frontend/pnpm-lock.yaml* ./

# Copy the scripts directory (needed for setup-pinecone)
COPY ./scripts/ /scripts/

# Update the setup-pinecone.js path in package.json
RUN sed -i 's|"setup-pinecone": "node ../scripts/setup-pinecone.js"|"setup-pinecone": "node /scripts/setup-pinecone.js"|g' package.json

# Install project dependencies (this layer will be cached if package files don't change)
# Use --no-frozen-lockfile as fallback if lockfile is missing or out of sync
RUN pnpm config set auto-install-peers true && \
    if [ -f pnpm-lock.yaml ]; then \
        echo "Lock file found, installing with frozen lockfile..." && \
        (pnpm install --no-optional --frozen-lockfile || pnpm install --no-optional --no-frozen-lockfile); \
    else \
        echo "No lock file found, installing without frozen lockfile..." && \
        pnpm install --no-optional --no-frozen-lockfile; \
    fi

# Copy the rest of the frontend files
COPY ./frontend/ ./

# Build the application
RUN pnpm build

# Expose the port the app runs on
EXPOSE 3000

# Start the application
CMD ["pnpm", "start"]