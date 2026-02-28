#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$ROOT_DIR/web"
DIST_DIR="$WEB_DIR/dist"

mkdir -p "$DIST_DIR"

GO_WASM_EXEC=""
for candidate in "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$(go env GOROOT)/misc/wasm/wasm_exec.js"; do
  if [[ -f $candidate ]]; then
    GO_WASM_EXEC="$candidate"
    break
  fi
done

if [[ -z $GO_WASM_EXEC ]]; then
  echo "wasm_exec.js not found under GOROOT" >&2
  exit 1
fi

GOOS=js GOARCH=wasm go build -trimpath -ldflags="-s -w" -o "$DIST_DIR/flashsr-kernel.wasm" "$ROOT_DIR/cmd/flashsr-wasm"
cp "$GO_WASM_EXEC" "$DIST_DIR/wasm_exec.js"
cp "$WEB_DIR/index.html" "$WEB_DIR/main.js" "$WEB_DIR/styles.css" "$DIST_DIR/"

echo "Built $DIST_DIR/flashsr-kernel.wasm"
echo "Copied web runtime and static assets into $DIST_DIR"
