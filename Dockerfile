# Global ARG scope for multi-arch build options
ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM
ARG TARGETARCH

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
FROM --platform=$BUILDPLATFORM rust:1.95-slim-bookworm AS rust-builder

ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH

# Install cross-compilers for both directions (x86_64 -> aarch64 and aarch64 -> x86_64)
# in the builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    build-essential \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    gcc-x86-64-linux-gnu \
    g++-x86-64-linux-gnu \
    libc6-dev-amd64-cross \
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

# Multi-arch compilation handling cross builds between amd64 and arm64 host platforms
RUN case "${TARGETPLATFORM}" in \
        "linux/amd64") \
            RUST_TARGET="x86_64-unknown-linux-gnu" ;; \
        "linux/arm64") \
            RUST_TARGET="aarch64-unknown-linux-gnu" ;; \
        *) \
            RUST_TARGET="$(rustc -vV | sed -n 's/host: //p')" ;; \
    esac && \
    if [ "$BUILDPLATFORM" != "$TARGETPLATFORM" ]; then \
        rustup target add "$RUST_TARGET" && \
        if [ "$TARGETARCH" = "arm64" ]; then \
            CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
            cargo build --release -p llmfit --target "$RUST_TARGET"; \
        elif [ "$TARGETARCH" = "amd64" ]; then \
            CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
            cargo build --release -p llmfit --target "$RUST_TARGET"; \
        else \
            cargo build --release -p llmfit --target "$RUST_TARGET"; \
        fi && \
        cp "target/${RUST_TARGET}/release/llmfit" target/release/llmfit; \
    else \
        cargo build --release -p llmfit; \
    fi

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
