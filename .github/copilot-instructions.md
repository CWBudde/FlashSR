# FlashSR-Go Copilot Instructions

## Developer workflow (use `just`)

- `just fmt`: formats via `treefmt` (Go: `gofumpt` + `gci`, plus `prettier`/`shfmt`).
- `just lint` / `just lint-fix`: runs `golangci-lint` (config in `.golangci.yml`).
- `just test` / `just test-race` / `just test-golden`.
- `just build`: builds the CLI to `bin/flashsr`.
- `just web-demo`: builds the WASM kernel and serves `web/dist`.

## Big picture: audio pipeline packages

- `flashsr/`: public API (`flashsr.New`, `Upsampler.Upsample16kTo48k`). Orchestrates model load → engine init → optional pre-resample → inference → peak normalize.
- `engine/`: backend interface (`engine.Engine`). `Run` must be safe for concurrent calls.
- `engine/ort/`: ONNX Runtime backend (via `github.com/yalue/onnxruntime_go`, cgo + dynamic `dlopen`). The ORT environment is initialized **once per process** (`sync.Once`), so set `FLASHSR_ORT_LIB` (or `ort.Config.LibraryPath`) before the first `ort.New`.
- `model/`: model bytes loader; priority is `FLASHSR_MODEL_PATH` → `model.Config.Path` → embedded `assets/model.onnx`. Optional SHA256 pin: `model.ExpectedSHA256`.
- `stream/`: streaming wrapper that matches upstream behavior (see constants in `stream/stream.go`: chunk `4000`, overlap `500`, output alignment skip `1000`, first-chunk trim `2000`, crossfade over `overlap*3`). Typical flow: `Write(...)` chunks, then `Flush()`, then drain with `Read(...)` (returns `io.EOF` when empty).
- `resample/`: sample-rate conversion for non-16kHz input. Prefer `resample.NewFor(...)` (fast linear, stateful) or `resample.NewPolyphase(...)` (FIR polyphase, quality options in `resample/polyphase.go`).
- `cmd/flashsr/`: Cobra CLI (`upsample`, `doctor`, `model download`), WAV I/O helpers live in `cmd/flashsr/wav.go`.
- `cmd/flashsr-wasm/` + `web/`: browser demo kernel compiled to Go/WASM (built by `web/build-wasm.sh`).

## Tests, fixtures, and build tags

- Default `go test ./...` is expected to pass without ONNX Runtime; ORT-dependent tests skip when `FLASHSR_ORT_LIB` is unset.
- Golden tests are gated by `//go:build golden` and use `.npy` fixtures in `internal/testutil/fixtures/`.
- Regenerate fixtures with Python (documented in `internal/testutil/fixtures/README.md`):
  - `FLASHSR_MODEL_PATH=/path/to/model.onnx python3 scripts/gen_fixtures.py --out internal/testutil/fixtures/`
- Golden comparisons use `internal/testutil.RMSError` and enforce RMS error thresholds (see `flashsr/golden_test.go`, `stream/golden_test.go`).

## Project conventions (non-generic)

- Audio values are float32 PCM in `[-1, 1]`. Call sites should clamp/validate inputs; `Upsampler` peak-normalizes output to `<= 0.999`.
- Public error “categories” are the sentinel errors in `flashsr/errors.go`; wrap underlying failures with `%w` so callers can match categories via `errors.Is`.
