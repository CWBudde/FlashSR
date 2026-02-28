#!/usr/bin/env python3
"""Generate golden test fixtures for FlashSR Go tests.

Usage:
    FLASHSR_MODEL_PATH=/path/to/model.onnx python3 scripts/gen_fixtures.py \
        [--out internal/testutil/fixtures/]
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("error: onnxruntime not installed (pip install onnxruntime)", file=sys.stderr)
    sys.exit(1)

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000
OVERLAP = 500
OUTPUT_OVERLAP_SKIP = 1000
FIRST_CHUNK_TRIM = 2000
UPSAMPLE_RATIO = 3


def sine(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def pink_noise(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float32)
    # Simple 1/f approximation via cumulative sum + high-pass mix.
    pink = np.cumsum(white)
    pink = pink - np.mean(pink)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink.astype(np.float32)


def sine_sweep(freq_start: float, freq_end: float, n: int) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    T = n / SAMPLE_RATE
    phase = 2 * np.pi * t * (freq_start + (freq_end - freq_start) * t / (2 * T))
    return np.sin(phase).astype(np.float32)


class BatchModel:
    def __init__(self, model_path: str):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, sess_opts)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_rank = len(self.sess.get_inputs()[0].shape)

    def run(self, pcm: np.ndarray) -> np.ndarray:
        if self.input_rank == 3:
            x = pcm.reshape(1, 1, -1)
        else:
            x = pcm.reshape(1, -1)
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        return out.flatten()


def stream_inference(model: BatchModel, pcm: np.ndarray) -> np.ndarray:
    """Replicate the upstream Python streaming algorithm."""
    overlap_buf = np.zeros(0, dtype=np.float32)
    prev_tail = np.zeros(0, dtype=np.float32)
    out_overlap = OVERLAP * UPSAMPLE_RATIO
    first_chunk = True
    output_parts = []

    i = 0
    while i < len(pcm):
        chunk = pcm[i:i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            # Zero-pad last chunk.
            chunk = np.concatenate([chunk, np.zeros(CHUNK_SIZE - len(chunk), dtype=np.float32)])

        model_input = np.concatenate([overlap_buf, chunk]).astype(np.float32)
        overlap_buf = chunk[-OVERLAP:].copy()

        raw = model.run(model_input)

        # Alignment skip.
        if len(raw) > OUTPUT_OVERLAP_SKIP:
            raw = raw[OUTPUT_OVERLAP_SKIP:]

        # Crossfade with previous tail.
        if len(prev_tail) > 0 and len(raw) > 0:
            n = min(out_overlap, len(prev_tail), len(raw))
            t = np.linspace(0, 1, n, dtype=np.float32)
            raw = raw.copy()
            raw[:n] = prev_tail[:n] * (1 - t) + raw[:n] * t

        if len(raw) >= out_overlap:
            prev_tail = raw[-out_overlap:].copy()
        else:
            prev_tail = raw.copy()

        if first_chunk:
            if len(raw) > FIRST_CHUNK_TRIM:
                raw = raw[FIRST_CHUNK_TRIM:]
            else:
                raw = np.zeros(0, dtype=np.float32)
            first_chunk = False

        output_parts.append(raw)
        i += CHUNK_SIZE

    if output_parts:
        return np.concatenate(output_parts)
    return np.zeros(0, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate FlashSR golden test fixtures")
    parser.add_argument("--out", default="internal/testutil/fixtures/",
                        help="Output directory for .npy files")
    args = parser.parse_args()

    model_path = os.environ.get("FLASHSR_MODEL_PATH")
    if not model_path:
        print("error: FLASHSR_MODEL_PATH env var not set", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = BatchModel(model_path)
    print(f"  input name={model.input_name!r}  rank={model.input_rank}")

    signals = {
        "sine": sine(440, SAMPLE_RATE),
        "pink": pink_noise(42, SAMPLE_RATE),
        "sweep": sine_sweep(50, 4000, SAMPLE_RATE),
    }

    # Save input signals so Go tests can load identical bytes (avoids RNG mismatch).
    for name, pcm in signals.items():
        path = os.path.join(args.out, f"in_{name}.npy")
        np.save(path, pcm)
        print(f"  saved input {path}  shape={pcm.shape}")

    for name, pcm in signals.items():
        print(f"Running batch inference on {name} ({len(pcm)} samples)...")
        out = model.run(pcm)
        path = os.path.join(args.out, f"ref_batch_{name}.npy")
        np.save(path, out)
        print(f"  saved {path}  shape={out.shape}")

    for name in ("sine", "pink"):
        pcm = signals[name]
        print(f"Running streaming inference on {name} ({len(pcm)} samples)...")
        out = stream_inference(model, pcm)
        path = os.path.join(args.out, f"ref_stream_{name}.npy")
        np.save(path, out)
        print(f"  saved {path}  shape={out.shape}")

    print("\nDone. Run golden tests with:")
    print(f"  FLASHSR_ORT_LIB=... go test -tags golden ./... -v")


if __name__ == "__main__":
    main()
