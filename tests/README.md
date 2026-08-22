# Test Suite Documentation

This directory houses integration, end-to-end, and infrastructure verification tests for the project.

## Structure & Organization

As the testing suite expands, new test scripts and configuration files should follow the established conventions under the `tests/` directory:

* `test_docker_compose.sh` - Integration test script for validating the Docker Compose services stack.
* `docker-compose.yml` - Compose configuration defining the test environment stack.
* `test_*.sh` (Future scripts) - Additional modular test scripts for specific subsystems or components.

## Running Existing Tests

### Docker Compose Integration Test
To execute the docker-compose integration test suite with BuildKit enabled, run the following from the project root:

```bash
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 ./tests/test_docker_compose.sh
```

The test script validates:

- Container endpoint reachability (/health)
- JSON payload validity from the embedded API (/api/v1/system)
- Static HTML UI asset serving (/) directly out of the compiled binary

---

## Adding Future Tests

When contributing new test scripts or environments to the `tests/` directory, please adhere to the following conventions and guidelines to ensure maintainability and smooth integration:

### 1. Naming Conventions
* Prefix test scripts clearly based on their scope (e.g., `test_cli.sh`, `test_api_auth.sh`, `test_migration.sh`).
* Keep any accompanying configuration files (like compose files or mock payloads) named relative to their target test or service.

### 2. Script Standards (`test_*.sh`)
* **Strict Error Handling**: Always include strict bash safety flags at the top of your script:
  ```bash
  set -euo pipefail
  ```
* **Path Independence**: Resolve the script's directory dynamically so tests can be reliably invoked from any path or wrapper:
  ```bash
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ```
* **Resource Cleanup**: Always use a trap function to guarantee temporary resources (containers, networks, files, volumes) are purged even if the test fails abruptly:
  ```bash
  cleanup() {
      echo "==> Cleaning up test resources..."
      # Add teardown logic here
  }
  trap cleanup EXIT
  ```
* **Consistent Output Styling**: Use clear section headers and check/cross markers to match existing reporting styles:
  ```bash
  echo "==> [1/2] Running sub-test..."
  echo "  ✔ Test condition met successfully"
  echo "  ✖ Test condition failed"
  exit 1
  ```

### 3. CI/CD Integration
Ensure new test scripts return an explicit non-zero exit code (`exit 1`) on failure so automated build pipelines and test runners immediately flag regressions.
