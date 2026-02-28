# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

FlashSR-Go is a Go library and CLI for audio super-resolution (16 kHz → 48 kHz) using an embedded ONNX model. It includes streaming inference, polyphase FIR resampling, a browser demo (Go/WASM), and a Pocket-TTS integration layer.

## Build & Development Commands

All commands use `just` (see `justfile`):

```bash
just build          # Build CLI to bin/flashsr
just test           # go test -v ./...
just test-race      # go test -race ./...
just test-golden    # go test -tags golden ./... (requires FLASHSR_ORT_LIB)
just bench          # go test -bench=. -benchmem -run=^$ ./...
just lint           # golangci-lint run (config: .golangci.yml)
just lint-fix       # golangci-lint run --fix
just fmt            # treefmt (gofmt + others)
just ci             # check-formatted + test + lint + check-tidy
just web-demo       # Build WASM kernel + serve on localhost:8080
```

Run a single test:

```bash
go test -v -run TestName ./flashsr/
```

Run golden tests (need ORT library + `.npy` fixtures):

```bash
export FLASHSR_ORT_LIB=/path/to/libonnxruntime.so
go test -tags golden ./...
```

Regenerate golden test fixtures:

```bash
FLASHSR_MODEL_PATH=assets/model.onnx uv run scripts/gen_fixtures.py --out internal/testutil/fixtures/
```

## Environment Variables

- `FLASHSR_ORT_LIB` — path to `libonnxruntime.so/.dylib/.dll` (required for ORT-backed inference)
- `FLASHSR_MODEL_PATH` — override embedded model path (optional)
- `HF_TOKEN` — Hugging Face token for `flashsr model download`

## Architecture

```
flashsr/          Public API: New(), Upsampler.Upsample16kTo48k(), Close()
                  Orchestrates: model load → engine init → optional pre-resample → inference → peak normalize
engine/           Engine interface (Run, Close, Info). Must be safe for concurrent calls.
engine/ort/       ONNX Runtime backend (cgo, dynamic dlopen via yalue/onnxruntime_go)
                  ORT env is initialized once per process (sync.Once) — set FLASHSR_ORT_LIB before first ort.New
model/            Model loader. Priority: FLASHSR_MODEL_PATH → Config.Path → embedded assets/model.onnx
                  SHA256 pin: model.ExpectedSHA256
stream/           Streaming wrapper matching upstream Python behavior
                  Constants: chunk 4000, overlap 500, output skip 1000, first-chunk trim 2000, crossfade overlap*3
                  Flow: Write() chunks → Flush() → drain with Read() (returns io.EOF)
resample/         Sample-rate conversion. Two implementations:
                  - NewFor(): fast linear (stateful, streaming)
                  - NewPolyphase(): FIR polyphase with quality presets (Fast/Balanced/Best)
                  FIR design vendored from cwbudde/algo-dsp (MIT)
pockettts/        Post-processor for Pocket-TTS output (24 kHz → 48 kHz via FlashSR)
assets/           Embedded ONNX model (~499 kB, go:embed)
cmd/flashsr/      Cobra CLI: upsample, doctor, model download. WAV I/O in wav.go
cmd/flashsr-wasm/ Browser demo kernel (GOOS=js GOARCH=wasm)
web/              Static frontend for WASM demo
internal/testutil Test utilities: .npy loader, signal generators, RMS error comparison
```

## Key Conventions

- Audio values are `float32` PCM in `[-1, 1]`. Output is peak-normalized to `≤ 0.999`.
- CGO_ENABLED=1 is required for ORT backend builds.
- Default `go test ./...` passes without ONNX Runtime — ORT-dependent tests skip when `FLASHSR_ORT_LIB` is unset.
- Golden tests use build tag `//go:build golden` and compare against Python reference `.npy` fixtures using RMS error thresholds.
- Public error categories are sentinel errors in `flashsr/errors.go`; wrap with `%w` for `errors.Is` matching.
- Linting uses `golangci-lint` v2 with `default: all` and selective disables. Magic numbers and long lines are allowed (math-heavy code). See `.golangci.yml` for full config.
- The `gosec` G115 check (integer overflow) is disabled — too noisy for audio sample conversions.
