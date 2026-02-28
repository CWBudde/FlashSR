# FlashSR-Go: Development Plan

## Comprehensive Plan for `github.com/MeKo-Christian/flashsr-go`

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

This document defines a phased plan for building **FlashSR-Go** — a production-ready Go
implementation of FlashSR audio super-resolution (16 kHz → 48 kHz) with ONNX Runtime,
an optional high-quality resampler via `algo-dsp`, and a Pocket-TTS integration path.

It is intentionally separated from:

- file container concerns (`wav`) and
- application orchestration concerns.

This plan is **actionable**: every phase contains **checkable tasks and subtasks**.

---

## Table of Contents

1. Project Scope and Goals
2. Repository and Module Boundaries
3. Architecture and Package Layout
4. API Design Principles
5. Phase Overview
6. Detailed Phase Plan (Phases 0–11)
7. Appendices
   - Appendix A: Testing and Validation Strategy
   - Appendix B: Benchmarking and Performance Strategy
   - Appendix C: Dependency and Versioning Policy
   - Appendix D: Release Engineering
   - Appendix E: Risks and Mitigations
   - Appendix F: Revision History

---

## 1. Project Scope and Goals

### 1.1 Primary Goals

- Provide a standalone Go library that performs audio super-resolution: Float32 PCM at
  16 kHz in, Float32 PCM at 48 kHz out.
- Deliver a battle-tested streaming mode with upstream-compatible overlap (500 samples),
  crossfade, and "first chunk" trimming — so artifact profiles match the Python reference.
- Ship a CLI (`flashsr`) that reads WAV, upsample, and writes 48 kHz WAV.
- Keep the ORT binding behind a clean `Engine` interface so future backends slot in without
  touching the public API.
- Embed the ONNX model at compile time so the binary is self-contained by default.

### 1.2 Included Scope (v1)

- `flashsr` public library: `New`, `Close`, `Upsample16kTo48k`, `Config`.
- `flashsr/engine` interface + `flashsr/engine/ort` default implementation via
  `yalue/onnxruntime_go` (cgo, dynamic `dlopen`).
- `flashsr/model` model loader: embedded ONNX + `--model-path` / `FLASHSR_MODEL_PATH`
  override.
- `flashsr/stream` streaming wrapper: ring buffer, overlap, crossfade, output generator.
- `cmd/flashsr` CLI: `upsample` subcommand (batch + streaming), `doctor` subcommand
  (ORT library check).

### 1.3 Included Scope (v1.1 / v2)

- `flashsr/resample` internal linear resampler for non-16 kHz inputs.
- `flashsr/resample` build-tag adapter for `github.com/cwbudde/algo-dsp/dsp/resample`
  (polyphase FIR, quality modes).
- Pocket-TTS post-processor integration.

### 1.4 Explicit Non-Goals

- GUI/visualization.
- Audio device APIs (ASIO/CoreAudio/JACK/PortAudio).
- File container codecs beyond minimal WAV reading/writing for the CLI.
- Multi-rate FlashSR model training or ONNX export.
- Pure-Go neural-net inference.

---

## 2. Repository and Module Boundaries

### 2.1 Ownership Model

- `github.com/MeKo-Christian/flashsr-go`: inference library + streaming + CLI.
- `github.com/cwbudde/algo-dsp`: polyphase FIR resampler (consumed via build tag, not
  hard-wired).
- `github.com/cwbudde/wav`: WAV I/O (consumed in CLI layer only).
- `github.com/yalue/onnxruntime_go`: ONNX Runtime cgo binding (default engine).

### 2.2 Boundary Rules

- No dependency on Wails/React/app-specific frameworks.
- No direct dependency on application logging/config frameworks.
- `flashsr` public API is transport-agnostic (PCM in, PCM out).
- WAV I/O lives only in `cmd/flashsr`, never in the library.
- `algo-dsp` dependency is gated behind build tag `algodsp` — default build has zero
  dependency on it.

---

## 3. Architecture and Package Layout

```plain
flashsr-go/
├── go.mod
├── README.md
├── PLAN.md
├── NOTICE
├── LICENSE
├── .golangci.yml
├── justfile
├── assets/
│   └── model.onnx              # embedded via go:embed (~499 kB, Apache-2.0)
├── flashsr/                    # Public library
│   ├── flashsr.go              # New, Close, Upsample16kTo48k, Config, Upsampler
│   ├── flashsr_test.go
│   └── errors.go
├── engine/                     # Engine interface + shared types
│   ├── engine.go               # Engine interface, EngineInfo
│   └── ort/
│       ├── ort.go              # ORT implementation (yalue/onnxruntime_go)
│       └── ort_test.go
├── model/
│   ├── model.go                # Loader: embedded + file override
│   └── model_test.go
├── stream/
│   ├── stream.go               # Streamer: ring buffer, overlap, crossfade
│   └── stream_test.go
├── resample/
│   ├── resample.go             # Resampler interface + linear default
│   ├── resample_algodsp.go     # build tag: algodsp
│   └── resample_test.go
├── internal/
│   └── testutil/               # WAV fixtures, float comparison helpers
└── cmd/
    └── flashsr/
        ├── main.go
        ├── cmd_upsample.go
        └── cmd_doctor.go
```

Notes:

- `internal/testutil` is test support only; never imported by the library.
- Stable APIs live in non-`internal` packages.
- The `engine/ort` package requires cgo and a path to the ORT shared library
  (`libonnxruntime.so` / `.dylib` / `.dll`). This is documented prominently.

---

## 4. API Design Principles

- Prefer small interfaces and concrete constructors.
- Deterministic behavior for same input/options (no hidden global state).
- Clear error semantics (`fmt.Errorf("flashsr: %w", err)`).
- Streaming-friendly APIs: `Streamer.Write([]float32) error` / `Streamer.Read([]float32) (int, error)`.
- Zero extra allocations on the hot inference path (reuse tensor memory).
- All public types and functions require doc comments.
- Numeric behaviour: input clamped to `[-1, 1]`; output normalized to ≤ 0.999 peak (matching Python upstream).

API shape:

```go
// Library
func New(cfg Config) (*Upsampler, error)
func (u *Upsampler) Upsample16kTo48k(x []float32) ([]float32, error)
func (u *Upsampler) Close() error

// Streaming
func NewStreamer(u *Upsampler, cfg StreamConfig) *Streamer
func (s *Streamer) Write(samples []float32) error
func (s *Streamer) Read(out []float32) (int, error)
func (s *Streamer) Flush() error
func (s *Streamer) Reset()

// Engine interface (internal use / advanced)
type Engine interface {
    Run(input []float32) ([]float32, error)
    Close() error
    Info() EngineInfo
}
```

---

## 5. Phase Overview

```plain
Phase 0:  Bootstrap & Governance                     [3 days]   ✅ Complete
Phase 1:  Engine Interface & ORT Binding             [5 days]   ✅ Complete
Phase 2:  Model Handling (embed + path override)     [2 days]   ✅ Complete
Phase 3:  Public Library & Batch Inference           [3 days]   ✅ Complete
Phase 4:  WAV I/O + CLI (upsample + doctor)          [4 days]   ✅ Complete
Phase 5:  Streaming (Buffer, Overlap, Crossfade)     [5 days]   ✅ Complete
Phase 6:  Golden Tests vs Python Reference           [4 days]   🔄 In Progress (all code done; Python fixtures pending)
Phase 7:  Benchmarks & Thread Tuning                 [3 days]   📋 Planned
Phase 8:  CI + Release Artifacts + Licensing         [4 days]   🔄 In Progress (CI workflow done; goreleaser + THIRD_PARTY_NOTICES pending)
Phase 9:  Linear Resampler (multi-rate v1.1)         [3 days]   🟡 Partial (resampler implemented; library/CLI wiring pending)
Phase 10: algo-dsp Resampler (build tag)             [3 days]   📋 Planned
Phase 11: Pocket-TTS Post-Processor Integration      [5 days]   📋 Planned
```

---

## 6. Detailed Phase Plan

### Phase 0: Bootstrap & Governance

**Status:** ✅ Complete.

**Done (condensed):**

- Bootstrapped module + repo governance: `go.mod`, `LICENSE`, `NOTICE`.
- Added local dev tooling: `justfile`, `.golangci.yml`, formatting/lint/test targets.
- Added CI workflow that runs tests and lint in a Go version matrix.

---

### Phase 1: Engine Interface & ORT Binding

**Status:** ✅ Complete.

**Done (condensed):**

- Implemented `engine.Engine` interface and ORT-backed engine in `engine/ort`.
- Added ORT env/session initialization guards and input/output tensor discovery.
- Covered with unit tests; integration smoke tests skip cleanly when ORT shared lib is missing.

---

### Phase 2: Model Handling

**Status:** ✅ Complete.

**Done (condensed):**

- Embedded the FlashSR ONNX model and added a loader with env/path overrides.
- Implemented `flashsr model download` with SHA256 verification and atomic writes.
- Added model tests (embedded/path/env/hash-mismatch).

---

### Phase 3: Public Library & Batch Inference

**Status:** ✅ Complete.

**Done (condensed):**

- Implemented the public `flashsr` package (`New`, `Close`, `Upsample16kTo48k`, config/errors).
- Batch path includes input clamping and output peak normalization.
- Tests are mock-backed (no ORT dependency) and validate basic invariants.

---

### Phase 4: WAV I/O + CLI

**Status:** ✅ Complete.

**Done (condensed):**

- Implemented `flashsr` CLI with `upsample` + `doctor` subcommands and WAV helpers.
- CLI tests cover WAV roundtrip + mock-engine pipeline; ORT-dependent tests skip via env guard.
- Streaming mode is wired behind a flag using the shared engine.

---

### Phase 5: Streaming Mode

**Status:** ✅ Complete.

**Done (condensed):**

- Implemented streaming wrapper with overlap/crossfade + first-chunk trimming.
- Added tests for buffering/flush/reset determinism and basic smoothness invariants.
- CLI supports a streaming mode flag.

---

### Phase 6: Golden Tests vs Python Reference

**Goal:** Verify that Go batch and streaming outputs match the Python upstream to within
numerical tolerance (RMS error ≤ −40 dB, no clipping, peak ≤ 1.0).

**Files:**

- Create: `internal/testutil/signals.go`
- Create: `internal/testutil/compare.go`
- Create: `internal/testutil/fixtures/` (WAV fixtures + reference outputs)

**Background:** Generate reference outputs from upstream Python:

```python
# upstream reference script (not part of this repo)
model = FASRONNX(model_path, ...)
out = model(input_pcm)
np.save("ref_batch.npy", out)
```

**Tasks:**

- [x] Write `internal/testutil` helpers
  - [x] `signals.go`: `Sine`, `SineSweep`, `PinkNoise`, `PeakAbs`
  - [x] `compare.go`: `RMSError` (dB), `RMS`, `HasNaNOrInf`
  - [x] `npy.go`: `LoadNPYFloat32` — minimal NPY v1/v2 reader for `<f4` arrays

- [ ] Generate Python reference fixtures (manual step — requires Python + onnxruntime)
  - [x] Created `scripts/gen_fixtures.py` — runs batch + streaming inference, saves `.npy`
  - [x] Documented regeneration steps in `internal/testutil/fixtures/README.md`
  - [ ] **TODO**: Run `python3 scripts/gen_fixtures.py` with real model to produce `.npy` files

- [x] Write golden tests (batch) — `flashsr/golden_test.go` (`//go:build golden`)
  - [x] `TestGolden_Batch_Sine` — 440 Hz sine, RMS error ≤ −40 dB
  - [x] `TestGolden_Batch_PinkNoise` — pink noise (seed=42)
  - [x] `TestGolden_Batch_SineSweep` — 50–4000 Hz sweep
  - [x] Skip gracefully: fixture not found → skip; `FLASHSR_ORT_LIB` not set → skip

- [x] Write golden tests (streaming) — `stream/golden_test.go` (`//go:build golden`)
  - [x] `TestGolden_Stream_Sine` — 4000-sample chunks, 5% length tolerance
  - [x] `TestGolden_Stream_PinkNoise`

- [x] Add property invariant tests (no build tag)
  - [x] `flashsr/property_test.go`: `TestProperty_NoNaN`, `TestProperty_PeakNormalized`, `TestProperty_OutputRate`
  - [x] `stream/property_test.go`: `TestProperty_Stream_NoNaN`, `TestProperty_Stream_OutputRate`
  - [x] All signals: 440 Hz sine, sine sweep 50–4000 Hz, pink noise (seed=42)

Exit criteria:

- [ ] `go test -tags golden ./... -v` all pass with RMS error ≤ −40 dB vs Python. ← pending fixture generation
- [x] `go test ./...` (without tag) all property tests pass.

---

### Phase 7: Benchmarks & Thread Tuning

**Goal:** Establish performance baselines; expose thread count knobs; document real-time factor.

**Files:**

- Create: `flashsr/bench_test.go`
- Create: `stream/bench_test.go`
- Create: `BENCHMARKS.md`

**Tasks:**

- [ ] Write benchmark suite

  ```go
  // flashsr/bench_test.go
  func BenchmarkUpsample_1s(b *testing.B)   // 16000 samples in
  func BenchmarkUpsample_10s(b *testing.B)  // 160000 samples in

  // stream/bench_test.go
  func BenchmarkStream_Chunk1000(b *testing.B)  // 62.5 ms @ 16kHz
  func BenchmarkStream_Chunk4000(b *testing.B)  // 250 ms @ 16kHz
  func BenchmarkStream_Chunk16000(b *testing.B) // 1s @ 16kHz
  ```

  - [ ] Add `b.ReportMetric(xRealtime, "x_realtime")` where:
        `xRealtime = (outputDuration / wallClock)`, `outputDuration = N * 3 / 48000`
  - [ ] Run: `go test -bench=. -benchmem ./... -run=^$ 2>&1 | tee BENCHMARKS.md`
  - [ ] Commit: `bench: add benchmarks and initial BENCHMARKS.md`

- [ ] Expose thread count in Config + CLI
  - [ ] `Config.NumThreadsIntra int` (default: 1, matches upstream)
  - [ ] `Config.NumThreadsInter int` (default: 1)
  - [ ] CLI flag: `--threads N` sets both
  - [ ] Commit: `feat: expose ORT thread count in Config and CLI`

- [ ] Run thread sweep and update BENCHMARKS.md
  - [ ] Test threads 1, 2, 4 on target machine
  - [ ] Document optimal setting for streaming (typically 1) vs batch (typically N_cpu)
  - [ ] Commit: `docs(bench): document thread sweep results`

Exit criteria:

- [ ] Benchmarks run without ORT panics.
- [ ] `BENCHMARKS.md` contains baseline numbers with machine info + Go version.
- [ ] Real-time factor ≥ 1.0 for `Chunk4000` on a 4-core modern CPU.

---

### Phase 8: CI + Release Artifacts + Licensing

**Status:** 🔄 In Progress.

**Done (condensed):**

- CI is in place and already runs: tests, race, vet, and `golangci-lint` (pinned), with golden tests gated behind an ORT secret/env.

**Remaining:**

- Add `THIRD_PARTY_NOTICES.md` with the exact attributions + pinned model hash.
- Add `.goreleaser.yml` and produce snapshot artifacts for the main target platforms.
- Verify release archives contain `NOTICE` + `THIRD_PARTY_NOTICES.md` and run `flashsr --help`.

Exit criteria:

- [x] `just ci` passes including race.
- [ ] GoReleaser snapshot produces valid archives.
- [ ] THIRD_PARTY_NOTICES.md is accurate and complete.

---

### Phase 9: Linear Resampler (Multi-Rate v1.1)

**Status:** 🟡 Partial.

**Done (condensed):**

- Added `resample` package with a stateful linear resampler and solid unit tests.
- Resampler supports streaming-style `Process(...)` calls without boundary jumps.

**Remaining (to make this actually usable end-to-end):**

- Wire resampling into the library (e.g., `flashsr.Config.InputRate` → resample to 16 kHz pre-inference).
- Expose the input rate in the CLI (e.g., `--input-rate 24000`) and verify the full WAV pipeline.

Exit criteria:

- [x] `go test ./resample/... -v` all pass.
- [ ] CLI accepts `--input-rate 24000` and produces correct 48 kHz output.

---

### Phase 10: algo-dsp Resampler (Build Tag)

**Goal:** High-quality polyphase FIR resampler via `algo-dsp` as an opt-in build tag,
keeping the zero-dependency default binary.

**Files:**

- Create: `resample/resample_algodsp.go` (`//go:build algodsp`)
- Create: `resample/resample_linear.go` (`//go:build !algodsp`)
- Modify: `resample/resample.go`

**Background:** `algo-dsp/dsp/resample` offers:

- `NewRational(up, down int, opts...Option) (*Resampler, error)`
- `WithQuality(QualityFast | QualityBalanced | QualityBest)`
- Profiles: Fast (16 taps, ~55 dB stopband), Balanced (32 taps, ~75 dB), Best (64 taps, ~90 dB)
- Operates on `float64`; we must convert `float32 ↔ float64`

**Tasks:**

- [ ] Refactor `resample.go` to have two build-tag files
  - [ ] `resample_linear.go` — `//go:build !algodsp` — wraps current linear impl
  - [ ] `resample_algodsp.go` — `//go:build algodsp` — wraps algo-dsp

- [ ] Write test that runs under both tags

  ```go
  // resample/resample_test.go (no build tag — runs always)
  func TestNewFor24kTo16k(t *testing.T) {
      r, err := NewFor(24000, 16000)
      require.NoError(t, err)
      // Same quality test as Phase 9
  }
  ```

- [ ] Implement algo-dsp adapter

  ```go
  //go:build algodsp

  package resample

  import (
      algoresample "github.com/cwbudde/algo-dsp/dsp/resample"
  )

  type algoDSPResampler struct {
      r       *algoresample.Resampler
      inRate  int
      outRate int
  }

  func newResampler(inRate, outRate int) (Resampler, error) {
      // Compute GCD reduction for up/down ratio
      // Create algoresample.NewRational(up, down, algoresample.WithQuality(algoresample.QualityBalanced))
      // Return &algoDSPResampler{...}
  }

  func (a *algoDSPResampler) Process(in []float32) ([]float32, error) {
      // Convert float32 → float64
      // r.Process(in64) → out64
      // Convert float64 → float32
  }
  ```

  - [ ] Add conditional dependency: `go get github.com/cwbudde/algo-dsp` (only needed with tag)
  - [ ] Commit: `feat(resample): add algo-dsp polyphase FIR resampler via build tag algodsp`

- [ ] Test with tag

  ```bash
  go test -tags algodsp ./resample/... -v
  ```

  - [ ] Assert tests pass
  - [ ] Assert `BenchmarkResample_24kTo16k` shows algo-dsp provides lower RMS noise vs linear
  - [ ] Commit: `test(resample): verify algo-dsp adapter matches interface contract`

- [ ] Add CLI flag `--resampler [linear|algodsp]` (shows error if binary not built with tag)
  - [ ] Commit: `feat(cmd): expose resampler backend selection in CLI`

Exit criteria:

- [ ] `go test ./resample/...` passes (linear, no tag).
- [ ] `go test -tags algodsp ./resample/...` passes (algo-dsp adapter).
- [ ] Neither build breaks `go vet ./...`.

---

### Phase 11: Pocket-TTS Post-Processor Integration

**Goal:** A `PostProcessor` interface in this repo that wraps the full pipeline:
Pocket-TTS WAV output (24 kHz) → resample → FlashSR → 48 kHz WAV. Useful for callers
that hold a `go-call-pocket-tts` `WAVResult`.

**Files:**

- Create: `pockettts/processor.go`
- Create: `pockettts/processor_test.go`

**Background:** `go-call-pocket-tts` returns `WAVResult{Data []byte, SampleRate int, ...}`.
`SampleRate` is 24000 Hz by default. PCM is int16 LE; we convert to float32.

**Tasks:**

- [ ] Define `PostProcessor` interface

  ```go
  // pockettts/processor.go
  package pockettts

  // PostProcessor takes raw PCM (any sample rate) and returns 48kHz upsampled PCM.
  type PostProcessor interface {
      Process(pcm []float32, inSampleRate int) (out []float32, outSampleRate int, err error)
  }

  // WAVResult mirrors go-call-pocket-tts WAVResult to avoid a hard dependency.
  type WAVResult struct {
      PCM        []float32
      SampleRate int
  }

  // ProcessWAVResult is a convenience wrapper.
  func ProcessWAVResult(p PostProcessor, r WAVResult) ([]float32, error)
  ```

  - [ ] Commit: `feat(pockettts): define PostProcessor interface`

- [ ] Write failing tests

  ```go
  func TestFlashSRPost_24kTo48k(t *testing.T) {
      u := requireUpsampler(t)
      proc := NewFlashSRProcessor(u, 24000)
      in := sinef32(440, 24000, 24000) // 1s @ 24kHz
      out, rate, err := proc.Process(in, 24000)
      require.NoError(t, err)
      assert.Equal(t, 48000, rate)
      assert.InDelta(t, 48000, len(out), 100)
  }
  ```

  - [ ] Run: `go test ./pockettts/... -v` → FAIL

- [ ] Implement `FlashSRProcessor`

  ```go
  type FlashSRProcessor struct {
      upsampler  *flashsr.Upsampler
      resampler  resample.Resampler // nil if inRate == 16000
  }

  func NewFlashSRProcessor(u *flashsr.Upsampler, inputSampleRate int) *FlashSRProcessor {
      // If inputSampleRate != 16000, build resample.NewFor(inputSampleRate, 16000)
  }

  func (p *FlashSRProcessor) Process(pcm []float32, inSampleRate int) ([]float32, int, error) {
      // 1. If inSampleRate != 16000: resample → 16kHz
      // 2. u.Upsample16kTo48k(pcm16k)
      // 3. Return (out, 48000, nil)
  }
  ```

  - [ ] Run: `go test ./pockettts/... -v` → PASS
  - [ ] Commit: `feat(pockettts): implement FlashSRProcessor for 24kHz→48kHz pipeline`

- [ ] Write helper: `Int16ToFloat32(pcm []int16) []float32`
  - [ ] Divides by 32768.0, clamps to [-1,1]
  - [ ] Write test: `TestInt16ToFloat32_Roundtrip`
  - [ ] Commit: `feat(pockettts): add Int16ToFloat32 PCM conversion helper`

- [ ] Add example
  ```go
  // pockettts/example_test.go
  func ExampleFlashSRProcessor() {
      // Show minimal pipeline: WAVResult → FlashSRProcessor → save 48kHz WAV
  }
  ```

  - [ ] Commit: `docs(pockettts): add runnable example`

Exit criteria:

- [ ] `go test ./pockettts/... -v` passes (skips without ORT lib).
- [ ] Integration: a caller can combine `go-call-pocket-tts` with `flashsr-go` in ~10 lines.

---

## 7. Roadmap (Gantt)

```mermaid
gantt
dateFormat  YYYY-MM-DD
title FlashSR-Go Roadmap (v1 → v2)

section v1 Core (Batch, 16k fixed)
Phase 0: Bootstrap                           :p0, 2026-03-02, 3d
Phase 1: Engine Interface + ORT Binding      :p1, after p0, 5d
Phase 2: Model Handling (embed + override)   :p2, after p1, 2d
Phase 3: Public Library + Batch Inference    :p3, after p2, 3d
Phase 4: WAV I/O + CLI                       :p4, after p3, 4d

section v1 Streaming
Phase 5: Streaming (Overlap/Crossfade)       :p5, after p4, 5d
Phase 6: Golden Tests vs Python Reference    :p6, after p5, 4d

section v1 Release Hardening
Phase 7: Benchmarks + Thread Tuning          :p7, after p6, 3d
Phase 8: CI + Release + Licensing            :p8, after p7, 4d

section v1.1 / v2 Enhancements
Phase 9: Linear Resampler (multi-rate)       :p9, after p8, 3d
Phase 10: algo-dsp Resampler (build tag)     :p10, after p9, 3d
Phase 11: Pocket-TTS Post-Processor          :p11, after p10, 5d
```

---

## Appendix A: Testing and Validation Strategy

### A.1 Test Types

- Unit tests (table-driven, edge-case heavy).
- Property/invariant tests (no NaN/Inf, peak ≤ 1.0, output rate = 3×).
- Golden vector tests (Go vs Python upstream, gated by `//go:build golden` tag).
- Integration tests across package boundaries.
- Race tests: `go test -race ./...` must always pass.

### A.2 Numerical Validation

- Tolerance policy: RMS error ≤ −40 dB vs Python reference (batch); ≤ −35 dB (streaming,
  allowing for minor alignment differences).
- Crossfade smoothness: no discontinuity > 0.05 between chunks.
- Peak normalization: output peak ∈ [0.0, 1.0] for any non-silent input.

### A.3 Coverage Targets

- `flashsr/`, `stream/`, `resample/`, `model/`: ≥ 85%.
- `engine/ort/`: ≥ 70% (many paths require ORT lib, skip-gated).

### A.4 Golden Test Fixtures

Fixtures are committed in `internal/testutil/fixtures/`:

- `sine_16k.wav` — 1 s, 440 Hz sine at 16 kHz
- `pinknoise_16k.wav` — 1 s pink noise at 16 kHz
- `sweep_16k.wav` — 2 s sweep 50 Hz→4 kHz at 16 kHz
- `ref_batch_sine.npy`, `ref_batch_noise.npy`, `ref_batch_sweep.npy` — Python batch outputs
- `ref_stream_sine.npy` — Python streaming output

Regeneration instructions live in `internal/testutil/fixtures/README.md`.

---

## Appendix B: Benchmarking and Performance Strategy

Maintain microbenchmarks for all hot paths. Key families:

| Benchmark               | Signal          | Expected Result (target)   |
| ----------------------- | --------------- | -------------------------- |
| `BenchmarkUpsample_1s`  | 16000 samp      | ≥ 2× realtime (any CPU)    |
| `BenchmarkUpsample_10s` | 160000 samp     | same realtime factor       |
| `BenchmarkStream_1000`  | 1000 samp/chunk | ≥ 1× realtime              |
| `BenchmarkStream_4000`  | 4000 samp/chunk | ≥ 5× realtime              |
| `BenchmarkResample_24k` | 24000→16000     | 0 extra allocs/op (linear) |

Track `allocs/op`, `bytes/op`, and `ns/op`. Update `BENCHMARKS.md` on each release with
date, Go version, and machine info.

---

## Appendix C: Dependency and Versioning Policy

- `yalue/onnxruntime_go`: pin ORT header version (≥ 1.24.1); update deliberately.
- `github.com/cwbudde/algo-dsp`: pin to `v0.x` tag; do not auto-upgrade.
- `github.com/cwbudde/wav`: pin to latest stable.
- Zero external dependencies on the hot inference path beyond ORT binding.
- `algo-dsp` must never appear in the default (no-tag) build graph.
- Minimum Go: 1.23 (embed, generics stable, slices package).

---

## Appendix D: Release Engineering

- Conventional commits for changelog generation.
- Tag-driven releases with GoReleaser.
- Pre-release channel (`v0.x`) until API freeze.
- Required release gates:
  - `golangci-lint` pass
  - `go test -race ./...` pass
  - `go test -tags golden ./...` pass (with ORT lib in CI secret)
  - BENCHMARKS.md baseline updated

**Distribution note:** The binary depends on `libonnxruntime` as a shared library.
Every release README must document:

1. Where to download the matching ORT release.
2. How to set `FLASHSR_ORT_LIB` or `LD_LIBRARY_PATH`.
3. How to run `flashsr doctor` to verify setup.

---

## Appendix E: Risks and Mitigations

| Risk                                               | Impact | Mitigation                                                                        |
| -------------------------------------------------- | ------ | --------------------------------------------------------------------------------- |
| ORT tensor name mismatch (`x` vs `audio_values`)   | High   | Introspect model metadata at session init; configurable fallback                  |
| ORT version/header mismatch in CI                  | Medium | Pin ORT version; document exact `libonnxruntime` version required                 |
| First-chunk / offset alignment differs from Python | High   | Golden tests with Python-generated fixtures; adjustable offset                    |
| cgo cross-compilation difficulties                 | Medium | Use per-OS native CI runners; document cross-compile limitations                  |
| algo-dsp API changes (v0.x)                        | Low    | Pin specific tag; build-tag isolation means breakage is silent until user opts in |
| FlashSR model license ambiguity                    | Medium | Verify Apache-2.0 on HF; include NOTICE with hash + source URL                    |
| ORT concurrency bugs on session init               | Low    | `sync.Once` for environment; serial session construction                          |

---

## Appendix F: Revision History

| Version | Date       | Author | Changes                                                            |
| ------- | ---------- | ------ | ------------------------------------------------------------------ |
| 0.1     | 2026-02-27 | Claude | Initial comprehensive plan from goal.md                            |
| 0.2     | 2026-02-27 | Claude | Marked completed scaffolding: Phases 0/3/5/9 ✅; Phases 1/2/4/8 🔄 |

---

This plan is a living document and should be updated after each phase completion and major
architectural decision.
