# Multi-stage build for llmfit
# Stage 1: Build the Rust binary
# rustc >= 1.95 required: sysinfo 0.39.x bumped its MSRV to 1.95.
# Pin the Debian release to match the runtime stage (bookworm). The default
# rust:1.95-slim base tracks trixie (glibc 2.39), which links the binary
# against symbols the bookworm runtime (glibc 2.36) does not provide, so the
# binary fails to start with "GLIBC_2.39 not found". Keep both stages on the
# same release so the linked glibc is always available at runtime.
FROM rust:1.95-slim-bookworm AS builder

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
