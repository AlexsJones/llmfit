# Building llmfit

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)

## Build

Build the default workspace members (`llmfit-core` and `llmfit-tui`):

```sh
cargo build --release
```

To build the desktop app as well:

```sh
cargo build --release --workspace
```

## Run

```sh
cargo run --release -p llmfit-tui
```

## Windows

### "This app can't run on your PC"

This message usually means one of the following:

- **Wrong architecture.** Download the **x64** build, not the ARM64 one, unless you are running an ARM64 device (for example, a Surface Pro X or Windows on ARM).
- **Missing WebView2 Runtime.** The desktop app requires the [Microsoft WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/). Install the Evergreen Standalone Installer, then relaunch the app.

If the problem persists after installing WebView2 and using the x64 build, please open an issue with your Windows version and CPU architecture.
