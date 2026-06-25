# llmfit-ffi: C-ABI Shared Library for llmfit-core

This crate provides a C-compatible Foreign Function Interface (FFI) for the `llmfit-core` library. It allows other languages (C, C++, Python, Pascal, C#, Go, etc.) to leverage llmfit's hardware detection and model fit analysis directly.

## Features

- **Stateful Context:** Manage hardware detection and model database state through an opaque handle.
- **JSON Exchange:** Uses JSON strings for data retrieval to avoid complex C-struct mapping and maintain forward compatibility.
- **Thread-Safe Errors:** Uses `thread_local` error storage to support multi-threaded applications.
- **Panic Safety:** Wraps entry points in `panic::catch_unwind` to prevent Rust panics from crossing the FFI boundary.

## Building

To build the shared library (`.so`, `.dll`, or `.dylib`):

```sh
cargo build -p llmfit-ffi --release
```

The output will be in `target/release/`.

## C-ABI Exported Functions

All functions use the `cdecl` calling convention.

### Lifecycle and State Management

| Function | Signature | Description |
| :--- | :--- | :--- |
| `llmfit_create_context` | `void* llmfit_create_context()` | Initializes the Rust core and returns an opaque handle. Returns `NULL` on failure. |
| `llmfit_destroy_context` | `void llmfit_destroy_context(void* ctx)` | Frees the context handle and all associated Rust memory. |

### Memory Management

| Function | Signature | Description |
| :--- | :--- | :--- |
| `llmfit_free_string` | `void llmfit_free_string(char* s)` | Frees a null-terminated UTF-8 string allocated by Rust. **Must be called** for every string returned by an `llmfit_*` function. |

### Error Handling

| Function | Signature | Description |
| :--- | :--- | :--- |
| `llmfit_get_last_error` | `char* llmfit_get_last_error()` | Returns the last error message for the current thread and clears it. Returns `NULL` if no error exists. |

### Data Retrieval

All retrieval functions return a **null-terminated UTF-8 string** (`char*`) containing JSON data, or `NULL` on failure.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `llmfit_version` | `char* llmfit_version()` | Returns the semantic version of the library. |
| `llmfit_get_system_info` | `char* llmfit_get_system_info(void* ctx)` | Returns a JSON object describing detected hardware. |
| `llmfit_recommend_models` | `char* llmfit_recommend_models(void* ctx, uint32_t limit)` | Returns a JSON array of model recommendations. `limit=0` returns all. |
| `llmfit_get_model_info` | `char* llmfit_get_model_info(void* ctx, const char* name)` | Returns a JSON array of fits for models matching the name substring. |

## Usage Example (C)

```c
#include <stdio.h>
#include <stdint.h>

// FFI Declarations
void* llmfit_create_context();
void  llmfit_destroy_context(void* ctx);
char* llmfit_recommend_models(void* ctx, uint32_t limit);
void  llmfit_free_string(char* s);
char* llmfit_get_last_error();

int main() {
    // 1. Create context
    void* ctx = llmfit_create_context();
    if (!ctx) {
        printf("Failed to create context\n");
        return 1;
    }

    // 2. Get top 5 recommendations
    char* json = llmfit_recommend_models(ctx, 5);
    if (json) {
        printf("Recommendations: %s\n", json);
        llmfit_free_string(json); // Always free strings!
    } else {
        char* err = llmfit_get_last_error();
        printf("Error: %s\n", err ? err : "Unknown");
        if (err) llmfit_free_string(err);
    }

    // 3. Cleanup
    llmfit_destroy_context(ctx);
    return 0;
}
```

## Maintenance Policy

1.  **SemVer:** Breaking changes to exports or JSON schemas require a Major version bump.
2.  **ABI Stability:** Use fixed-width types (e.g., `uint32_t`) for all numeric arguments.
3.  **JSON Forward Compatibility:** Clients should ignore unknown fields in JSON responses.
