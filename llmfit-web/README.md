# llmfit-web

React + Vite frontend for the llmfit local web dashboard.

## Development

```sh
npm ci
npm run dev
```

This starts Vite on `http://127.0.0.1:5173` and proxies `/api/*` to `http://127.0.0.1:8787`.

## Build

```sh
npm run build
```

Build output is written to `llmfit-web/dist` and embedded into `llmfit serve` at compile time.

## Pointing at a self-hosted backend

By default the frontend talks to the same origin (the Vite dev proxy forwards
`/api` to `http://127.0.0.1:8787`). To point the built frontend at a remote
`llmfit serve` instance (LAN IP, tunnel URL, or cloud backend), set `VITE_API_BASE`
before building:

```sh
# copy and edit .env.example -> .env
echo 'VITE_API_BASE=https://llmfit-xyz.trycloudflare.com' > .env
npm run build
```

`VITE_API_BASE` accepts a bare origin (trailing slashes are trimmed). The value
is baked into the bundle at build time, so rebuild after changing it.
