#!/usr/bin/env bash
# Automated model database update script for llmfit
# This script:
# 1. Runs the HuggingFace model scraper to fetch latest model data
# 2. Verifies the JSON output is valid
# 3. Rebuilds the Rust binary with updated embedded data
# 4. Optionally runs tests to ensure everything works

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_FILE="$PROJECT_ROOT/data/hf_models.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  llmfit Model Database Update${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Error: python3 not found${NC}"
    exit 1
fi

# Backup existing data file
if [ -f "$DATA_FILE" ]; then
    BACKUP_FILE="$DATA_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}📦 Backing up existing data to:${NC}"
    echo "   $BACKUP_FILE"
    cp "$DATA_FILE" "$BACKUP_FILE"
    echo
fi

# Run the scraper
echo -e "${BLUE}🔄 Running HuggingFace model scraper...${NC}"
if [ "$#" -gt 0 ]; then
    echo -e "${BLUE}   Scraper args:${NC} $*"
fi
echo
cd "$PROJECT_ROOT"
python3 scripts/scrape_hf_models.py "$@"

if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}✗ Scraper failed${NC}"
    exit 1
fi

echo

# Verify JSON is valid
echo -e "${BLUE}🔍 Verifying JSON output...${NC}"
if ! python3 -m json.tool "$DATA_FILE" > /dev/null 2>&1; then
    echo -e "${RED}✗ Invalid JSON generated${NC}"
    # Restore backup if available
    if [ -f "$BACKUP_FILE" ]; then
        echo -e "${YELLOW}📦 Restoring backup...${NC}"
        mv "$BACKUP_FILE" "$DATA_FILE"
    fi
    exit 1
fi

MODEL_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_FILE'))))")
echo -e "${GREEN}✓ Valid JSON with $MODEL_COUNT models${NC}"
echo

# Check if cargo is available
if command -v cargo &> /dev/null; then
    # Rebuild with updated data
    echo -e "${BLUE}🔨 Rebuilding llmfit with updated model data...${NC}"
    cargo build --release

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful${NC}"
        echo

        # Show build artifact location
        if [ -f "$PROJECT_ROOT/target/release/llmfit" ]; then
            BINARY_SIZE=$(ls -lh "$PROJECT_ROOT/target/release/llmfit" | awk '{print $5}')
            echo -e "${GREEN}📦 Binary location:${NC} target/release/llmfit (${BINARY_SIZE})"
        fi
    else
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ cargo not found, skipping rebuild${NC}"
    echo -e "${YELLOW}  Run 'cargo build --release' manually to rebuild${NC}"
fi

echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Model database update complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "  • Run './target/release/llmfit' to test the updated binary"
echo "  • Check 'data/hf_models.json' for the updated model list"
echo "  • Example: ./scripts/update_models.sh --threads 8 --gguf-sources"
if [ ! -z "$BACKUP_FILE" ]; then
    echo "  • Delete backup file if satisfied: rm $BACKUP_FILE"
fi
echo