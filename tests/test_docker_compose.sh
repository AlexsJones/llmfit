#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory for execution path independence
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"
LLMFIT_URL="${LLMFIT_URL:-http://localhost:8787}"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "==> Starting Docker Compose stack using ${COMPOSE_FILE}..."
docker compose -f "${COMPOSE_FILE}" up -d --build

cleanup() {
  echo "==> Tearing down Docker Compose stack..."
  docker compose -f "${COMPOSE_FILE}" down -v
}
trap cleanup EXIT

wait_for_service() {
  local url="$1"
  local description="$2"
  local retries=0

  echo "==> Waiting for ${description} at ${url}..."
  until curl -s -f "${url}" > /dev/null 2>&1; do
    retries=$((retries + 1))
    if [ "${retries}" -ge "${MAX_RETRIES}" ]; then
      echo "✖ ERROR: Timed out waiting for ${description} at ${url}"
      docker compose -f "${COMPOSE_FILE}" logs
      exit 1
    fi
    sleep "${RETRY_INTERVAL}"
  done
  echo "✔ ${description} is up and responding."
}

# 1. Verify container health endpoint
wait_for_service "${LLMFIT_URL}/health" "Health Check Endpoint"

# 2. Verify backend system API response payload
echo "==> Verifying System API payload..."
API_RESPONSE=$(curl -s -S -f "${LLMFIT_URL}/api/v1/system")

if echo "${API_RESPONSE}" | grep -q '"system"'; then
  echo "✔ System API check passed via ${LLMFIT_URL}/api/v1/system"
else
  echo "✖ ERROR: System API response did not contain expected payload."
  echo "Received: ${API_RESPONSE}"
  docker compose -f "${COMPOSE_FILE}" logs llmfit
  exit 1
fi

# 3. Verify static HTML asset serving (embedded web UI)
echo "==> Verifying embedded Web UI static asset serving..."
HTML_RESPONSE=$(curl -s -S -f "${LLMFIT_URL}/")

if echo "${HTML_RESPONSE}" | grep -i -q '<!DOCTYPE html>'; then
  echo "✔ Static Web UI index.html successfully served by Rust backend"
else
  echo "✖ ERROR: Index route did not return expected HTML structure."
  echo "Received: ${HTML_RESPONSE}"
  docker compose -f "${COMPOSE_FILE}" logs llmfit
  exit 1
fi

echo "==> All integration tests passed successfully!"
