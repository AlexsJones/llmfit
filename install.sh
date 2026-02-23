#!/bin/sh
# llmfit installer
# Usage: curl -fsSL https://raw.githubusercontent.com/AlexsJones/llmfit/main/install.sh | sh
#        curl -fsSL ... | sh -s -- --local   # Install to ~/.local/bin (no sudo)
#
# Downloads the latest llmfit release from GitHub and installs
# the binary to /usr/local/bin (or ~/.local/bin with --local or if no sudo).

set -e

REPO="AlexsJones/llmfit"
BINARY="llmfit"
LOCAL_INSTALL=""

# --- helpers ---

info() { printf '  \033[1;34m>\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m>\033[0m %s\n' "$*"; }
err()  { printf '  \033[1;31m!\033[0m %s\n' "$*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || err "Required tool '$1' not found. Please install it and try again."
}

# --- parse arguments ---

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --local|-l)
                LOCAL_INSTALL="1"
                ;;
            --help|-h)
                echo "Usage: install.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --local, -l    Install to ~/.local/bin (no sudo required)"
                echo "  --help, -h     Show this help message"
                exit 0
                ;;
            *)
                warn "Unknown option: $1"
                ;;
        esac
        shift
    done
}

# --- detect platform ---

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)  OS="unknown-linux-musl" ;;
        Darwin) OS="apple-darwin" ;;
        *)      err "Unsupported OS: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64)   ARCH="x86_64" ;;
        aarch64|arm64)  ARCH="aarch64" ;;
        *)              err "Unsupported architecture: $ARCH" ;;
    esac

    PLATFORM="${ARCH}-${OS}"
}

# --- fetch latest release ---

fetch_latest_tag() {
    need curl
    need tar

    # Use the releases redirect instead of the API to avoid GitHub's
    # 60-request/hour rate limit on unauthenticated API calls (403).
    TAG="$(curl -fsSI "https://github.com/${REPO}/releases/latest" 2>/dev/null \
        | grep -i '^location:' \
        | head -1 \
        | sed 's|.*/tag/||' \
        | tr -d '\r\n')"

    [ -n "$TAG" ] || err "Could not determine latest release. Check https://github.com/${REPO}/releases"
}

# --- download and install ---

install() {
    ASSET="${BINARY}-${TAG}-${PLATFORM}.tar.gz"
    URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR"' EXIT

    info "Downloading ${BINARY} ${TAG} for ${PLATFORM}..."
    curl -fsSL "$URL" -o "${TMPDIR}/${ASSET}" \
        || err "Download failed. Asset '${ASSET}' may not exist for your platform.\n  Check: https://github.com/${REPO}/releases/tag/${TAG}"

    info "Extracting..."
    tar -xzf "${TMPDIR}/${ASSET}" -C "$TMPDIR"

    # Find the binary in the extracted contents
    BIN="$(find "$TMPDIR" -name "$BINARY" -type f | head -1)"
    [ -n "$BIN" ] || err "Binary not found in archive. Release asset may have an unexpected layout."
    chmod +x "$BIN"

    # Determine install directory
    if [ -n "$LOCAL_INSTALL" ]; then
        # User explicitly requested local install
        INSTALL_DIR="${HOME}/.local/bin"
        mkdir -p "$INSTALL_DIR"
        info "Installing to ${INSTALL_DIR} (--local mode)..."
    elif [ -w /usr/local/bin ]; then
        # /usr/local/bin is writable without sudo
        INSTALL_DIR="/usr/local/bin"
    elif command -v sudo >/dev/null 2>&1 && [ -t 0 ]; then
        # sudo is available and we're in an interactive terminal
        info "Installing to /usr/local/bin (requires sudo)..."
        if sudo mv "$BIN" "/usr/local/bin/${BINARY}"; then
            info "Installed ${BINARY} to /usr/local/bin/${BINARY}"
            return
        else
            warn "sudo failed, falling back to ~/.local/bin"
            INSTALL_DIR="${HOME}/.local/bin"
            mkdir -p "$INSTALL_DIR"
        fi
    else
        # No write access and no interactive sudo, use local install
        INSTALL_DIR="${HOME}/.local/bin"
        mkdir -p "$INSTALL_DIR"
        info "Installing to ${INSTALL_DIR} (no sudo available)..."
    fi

    mv "$BIN" "${INSTALL_DIR}/${BINARY}"
    info "Installed ${BINARY} to ${INSTALL_DIR}/${BINARY}"

    # Check if install dir is in PATH
    case ":$PATH:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            warn "Add ${INSTALL_DIR} to your PATH to use '${BINARY}' directly:"
            echo ""
            echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
            echo ""
            ;;
    esac
}

# --- main ---

main() {
    parse_args "$@"
    info "llmfit installer"
    detect_platform
    fetch_latest_tag
    install
    info "Done. Run '${BINARY}' to get started."
}

main "$@"
