#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-tests/docker-compose.yml}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:8787}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8788}"
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
      echo "ERROR: Timed out waiting for ${description} at ${url}"
      docker compose -f "${COMPOSE_FILE}" logs
      exit 1
    fi
    sleep "${RETRY_INTERVAL}"
  done
  echo "✔ ${description} is up and responding."
}

# 1. Verify direct backend service availability
wait_for_service "${BACKEND_URL}/api/v1/system" "Direct Backend API"

# 2. Verify frontend service availability
wait_for_service "${FRONTEND_URL}" "Frontend UI"

# 3. Verify end-to-end API request through frontend Vite proxy
echo "==> Verifying end-to-end API proxy (Frontend -> Backend)..."
PROXY_RESPONSE=$(curl -s -S -f "${FRONTEND_URL}/api/v1/system")

if echo "${PROXY_RESPONSE}" | grep -q '"system"'; then
  echo "✔ End-to-end API proxy check passed via ${FRONTEND_URL}/api/v1/system"
else
  echo "ERROR: Proxy response did not contain expected payload."
  echo "Received: ${PROXY_RESPONSE}"
  docker compose -f "${COMPOSE_FILE}" logs llmfit-frontend
  exit 1
fi

echo "==> All integration tests passed successfully!"
