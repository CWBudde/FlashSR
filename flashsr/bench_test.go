package flashsr_test

import (
	"math"
	"os"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
)

// requireOrtUpsamplerB creates a real ORT-backed Upsampler for benchmarks.
// Skips when FLASHSR_ORT_LIB is not set.
func requireOrtUpsamplerB(b *testing.B) *flashsr.Upsampler {
	b.Helper()

	ortLib := os.Getenv("FLASHSR_ORT_LIB")
	if ortLib == "" {
		b.Skip("FLASHSR_ORT_LIB not set; skipping ORT benchmark")
	}

	u, err := flashsr.New(flashsr.Config{ORTLibPath: ortLib, NumThreadsIntra: 1, NumThreadsInter: 1})
	if err != nil {
		b.Fatalf("flashsr.New: %v", err)
	}

	b.Cleanup(func() { _ = u.Close() })

	return u
}

func makeBenchSine(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
	}

	return out
}

// BenchmarkUpsample_1s benchmarks a single 1-second (16000-sample) batch inference.
func BenchmarkUpsample_1s(b *testing.B) {
	u := requireOrtUpsamplerB(b)
	input := makeBenchSine(16000)

	b.ResetTimer()

	for b.Loop() {
		out, err := u.Upsample16kTo48k(input)
		if err != nil {
			b.Fatalf("Upsample16kTo48k: %v", err)
		}

		// Report x_realtime: output duration / wall-clock time per op.
		// Output duration = len(out) / 48000 s; b.Elapsed() gives total time.
		_ = out
	}

	// After the loop we can compute the real-time factor.
	outDur := float64(len(input)*3) / 48000.0 // seconds of audio output per call
	nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
	xRealtime := outDur / (nsPerOp / 1e9)
	b.ReportMetric(xRealtime, "x_realtime")
}

// BenchmarkUpsample_10s benchmarks a 10-second (160000-sample) batch inference.
func BenchmarkUpsample_10s(b *testing.B) {
	u := requireOrtUpsamplerB(b)
	input := makeBenchSine(160000)

	b.ResetTimer()

	for b.Loop() {
		out, err := u.Upsample16kTo48k(input)
		if err != nil {
			b.Fatalf("Upsample16kTo48k: %v", err)
		}

		_ = out
	}

	outDur := float64(len(input)*3) / 48000.0
	nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
	xRealtime := outDur / (nsPerOp / 1e9)
	b.ReportMetric(xRealtime, "x_realtime")
}
