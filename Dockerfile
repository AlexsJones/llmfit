# Global ARG scope for FROM line
ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM
ARG TARGETARCH

# =========================================================
# STAGE 1: Build Rust llmfit Executable
# =========================================================
FROM --platform=$BUILDPLATFORM rust:1.95-slim-bookworm AS rust-builder

# Redeclare global ARGs inside stage scope
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    build-essential \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy all workspace members
COPY llmfit-core/ ./llmfit-core/
COPY llmfit-tui/ ./llmfit-tui/
COPY llmfit-desktop/ ./llmfit-desktop/

# Fast native build if target matches build host; cross-compile only when targeting arm64 from amd64
RUN if [ "$BUILDPLATFORM" != "$TARGETPLATFORM" ] && [ "$TARGETARCH" = "arm64" ]; then \
        rustup target add aarch64-unknown-linux-gnu && \
        CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
        cargo build --release -p llmfit --target aarch64-unknown-linux-gnu && \
        cp target/aarch64-unknown-linux-gnu/release/llmfit target/release/llmfit; \
    else \
        cargo build --release -p llmfit; \
    fi

# =========================================================
# STAGE 2: Build Web UI Static Assets
# =========================================================
FROM node:20-slim AS web-builder

WORKDIR /app/llmfit-web
COPY llmfit-web/package*.json ./
RUN npm ci

COPY llmfit-web/ ./
RUN npm run build

# =========================================================
# STAGE 3: Unified Runtime Container
# =========================================================
FROM node:20-slim

WORKDIR /app

# Install system utilities for GPU/hardware introspection and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    pciutils \
    lshw \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=rust-builder /build/target/release/llmfit /usr/local/bin/llmfit

# Copy built Web UI assets
COPY --from=web-builder /app/llmfit-web /app/llmfit-web

# Setup non-root user by renaming default 'node' user (UID 1000) and set permissions
RUN usermod -l llmfit node && \
    groupmod -n llmfit node && \
    usermod -d /home/llmfit -m llmfit && \
    mkdir -p /tmp/.npm && \
    chmod -R 777 /tmp && \
    chown -R llmfit:llmfit /app /usr/local/bin/llmfit

WORKDIR /app/llmfit-web

# Entrypoint script handling CLI vs Web UI execution
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

USER llmfit
EXPOSE 8787

# Set default command to output JSON recommendations
# In Kubernetes, this will run once per node and log results
ENTRYPOINT ["/app/entrypoint.sh"]
