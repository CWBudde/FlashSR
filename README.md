# FlashSR Go

> **Upstream / Original Project:** [ysharma3501/FlashSR](https://github.com/ysharma3501/FlashSR)
>
> This Go implementation is based on and inspired by the original FlashSR project.

`flashsr-go` is a Go library and CLI for FlashSR audio super-resolution.
It runs the FlashSR ONNX model to convert speech-like audio to 48 kHz output.

- Library API for in-process audio pipelines.
- CLI for WAV input/output workflows.
- Streaming wrapper with overlap/crossfade behavior.
- Optional resampling for non-16 kHz inputs.
- Browser demo kernel compiled to Go/WASM.

## Requirements

- Go `1.25+`
- ONNX Runtime shared library (`libonnxruntime`)
  - Linux: `.so`
  - macOS: `.dylib`
  - Windows: `.dll`

FlashSR uses `github.com/yalue/onnxruntime_go`, which loads ONNX Runtime dynamically.
Set the library path with `FLASHSR_ORT_LIB` or `--ort-lib`.

## Quick Start (CLI)

Build:

```bash
go build -o flashsr ./cmd/flashsr
```

Set ONNX Runtime path:

```bash
export FLASHSR_ORT_LIB=/path/to/libonnxruntime.so
```

Run environment checks:

```bash
./flashsr doctor
```

Upsample a WAV file:

```bash
./flashsr upsample --input in.wav --output out_48k.wav
```

Use explicit input rate override:

```bash
./flashsr upsample \
  --input in.wav \
  --output out_48k.wav \
  --input-rate 24000
```

Use streaming mode:

```bash
./flashsr upsample \
  --input in.wav \
  --output out_48k.wav \
  --stream \
  --chunk-size 4000
```

## CLI Commands

- `flashsr upsample`: WAV -> 48 kHz WAV
- `flashsr doctor`: model + ONNX Runtime diagnostics
- `flashsr model download`: fetch model from Hugging Face

Download model manually:

```bash
./flashsr model download --out assets/model.onnx
```

Useful environment variables:

- `FLASHSR_ORT_LIB`: path to `libonnxruntime`
- `FLASHSR_MODEL_PATH`: override embedded model file
- `HF_TOKEN`: Hugging Face token for gated/private model access

## Library Usage

```go
package main

import (
	"os"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
)

func main() {
	u, err := flashsr.New(flashsr.Config{
		ORTLibPath: os.Getenv("FLASHSR_ORT_LIB"),
		InputRate:  24000, // auto-resample to 16 kHz before inference
	})
	if err != nil {
		panic(err)
	}
	defer u.Close()

	in := []float32{0, 0.1, -0.1} // example PCM
	out, err := u.Upsample16kTo48k(in)
	if err != nil {
		panic(err)
	}

	_ = out
}
```

## Web Demo (Go/WASM)

The repository includes a browser demo in `web/` with an explicit wasm kernel in
`cmd/flashsr-wasm`.

Behavior:

- Upload a WAV file.
- Analyze sample rate quality.
- If input sample rate is below `48 kHz`, resample to `48 kHz`.
- If input sample rate is `>= 48 kHz`, keep the original WAV bytes.

Local run:

```bash
./web/build-wasm.sh
python3 -m http.server 8080 -d web/dist
```

or:

```bash
just web-demo
```

Open: <http://localhost:8080>

GitHub Pages deploy is defined in:

- `.github/workflows/deploy-pages.yml`

It builds `web/dist/flashsr-kernel.wasm`, copies `wasm_exec.js`, and uploads
`web/dist` as the Pages artifact.

## Development

Run tests:

```bash
go test ./...
```

Useful `just` targets:

- `just test`
- `just lint`
- `just ci`
- `just web-demo`

## Notes

- Embedded model: `assets/model.onnx`
- Pinned model hash is `model.ExpectedSHA256` in `model/model.go`.
- The ORT backend currently reports provider `CPU`.

## License

MIT license for this repository code.

See:

- `LICENSE`
- `NOTICE`
- `THIRD_PARTY_NOTICES.md`
