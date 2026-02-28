# Golden Test Fixtures

This directory holds the reference `.npy` files used by the `golden`-tagged tests.
They are generated from the upstream Python `FlashSR` implementation so that the
Go port can be validated against bit-accurate reference outputs.

## Files

| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
| `ref_batch_sine.npy`  | Batch output for 1 s 440 Hz sine at 16 kHz            |
| `ref_batch_pink.npy`  | Batch output for 1 s pink noise (seed=42) at 16 kHz   |
| `ref_batch_sweep.npy` | Batch output for 1 s sine sweep 50–4000 Hz at 16 kHz  |
| `ref_stream_sine.npy` | Streaming output (4000-sample chunks) for sine signal |
| `ref_stream_pink.npy` | Streaming output (4000-sample chunks) for pink noise  |

## Regenerating Fixtures

Prerequisites:

```bash
pip install onnxruntime numpy scipy
# and have the FlashSR ONNX model available
```

Run the generation script from the repository root:

```bash
FLASHSR_MODEL_PATH=/path/to/model.onnx \
  python3 scripts/gen_fixtures.py \
  --out internal/testutil/fixtures/
```

## Running Golden Tests

```bash
# Requires FLASHSR_ORT_LIB and the fixture files above
FLASHSR_ORT_LIB=/usr/lib/libonnxruntime.so \
  go test -tags golden ./... -v
```
