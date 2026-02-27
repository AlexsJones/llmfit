# Multi-stage build for llmfit
# Stage 1: Build the Rust binary
FROM rust:1.85-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy all workspace members
COPY llmfit-core/ ./llmfit-core/
COPY llmfit-tui/ ./llmfit-tui/
COPY llmfit-desktop/ ./llmfit-desktop/
COPY data/ ./data/

# Build release binary for llmfit-tui
RUN cargo build --release -p llmfit

# Stage 2: Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies for hardware detection
RUN apt-get update && apt-get install -y \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /build/target/release/llmfit /usr/local/bin/llmfit

# Create a non-root user
RUN useradd -m -u 1000 llmfit && \
    chown -R llmfit:llmfit /usr/local/bin/llmfit

USER llmfit

# Set default command to output JSON recommendations
# In Kubernetes, this will run once per node and log results
ENTRYPOINT ["/usr/local/bin/llmfit"]
CMD ["recommend", "--json"]
