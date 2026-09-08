# =========================================================
# STAGE 1: Build Web UI Static Assets
# =========================================================
FROM node:20-slim AS web-builder

WORKDIR /app/llmfit-web
COPY llmfit-web/package*.json ./
RUN npm ci

COPY llmfit-web/ ./
RUN npm run build

# =========================================================
# STAGE 2: Build Rust llmfit Executable
# =========================================================
# rustc >= 1.95 required: sysinfo 0.39.x bumped its MSRV to 1.95.
# Pin the Debian release to match the runtime stage (bookworm). The default
# rust:1.95-slim base tracks trixie (glibc 2.39), which links the binary
# against symbols the bookworm runtime (glibc 2.36) does not provide, so the
# binary fails to start with "GLIBC_2.39 not found". Keep both stages on the
# same release so the linked glibc is always available at runtime.
#
# Each platform is built natively: the Docker workflow runs one job per
# platform on a matching runner, so the builder image is always the host's
# own architecture. Do not pin `--platform=$BUILDPLATFORM` here. Declaring
# BUILDPLATFORM with a default overrode the value buildx supplies, which
# pulled the amd64 toolchain onto the arm64 runner and ran rustc under QEMU
# user-mode emulation, where it segfaults (v1.1.13 release failure).
FROM rust:1.95-slim-bookworm AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Copy built frontend dist into workspace so rust-embed includes it at compile-time
COPY --from=web-builder /app/llmfit-web/dist ./llmfit-web/dist

# Build the release binary for the host architecture
RUN cargo build --release -p llmfit

# =========================================================
# STAGE 3: Unified Runtime Container
# =========================================================
FROM debian:bookworm-slim

# Install runtime dependencies for hardware detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    pciutils \
    lshw \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=rust-builder /build/target/release/llmfit /usr/local/bin/llmfit

# Create non-root user
RUN useradd -m -u 1000 llmfit && \
    chown -R llmfit:llmfit /usr/local/bin/llmfit

USER llmfit
EXPOSE 8787

# Set default command to output JSON recommendations
# In Kubernetes, this will run once per node and log results
ENTRYPOINT ["/usr/local/bin/llmfit"]
CMD ["recommend", "--json"]
