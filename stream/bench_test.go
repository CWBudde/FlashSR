package stream_test

import (
	"math"
	"os"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/stream"
)

// requireOrtUpsamplerForStream creates a real ORT-backed Upsampler for stream benchmarks.
// Skips when FLASHSR_ORT_LIB is not set.
func requireOrtUpsamplerForStream(b *testing.B) *flashsr.Upsampler {
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

func makeBenchStreamSine(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
	}

	return out
}

func runStreamBench(b *testing.B, chunkSize int) {
	b.Helper()

	u := requireOrtUpsamplerForStream(b)

	const totalSamples = 16000 // 1 s @ 16 kHz
	input := makeBenchStreamSine(totalSamples)

	outBuf := make([]float32, totalSamples*4) // generous upper bound

	b.ResetTimer()

	for b.Loop() {
		st := stream.New(u.Engine(), stream.Config{ChunkSize: chunkSize})

		for i := 0; i < len(input); i += chunkSize {
			end := min(i+chunkSize, len(input))

			err := st.Write(input[i:end])
			if err != nil {
				b.Fatalf("Write: %v", err)
			}
		}

		err := st.Flush()
		if err != nil {
			b.Fatalf("Flush: %v", err)
		}

		n, err := st.Read(outBuf)
		if err != nil {
			b.Fatalf("Read: %v", err)
		}

		_ = outBuf[:n]

		st.Reset()
	}

	outDur := float64(totalSamples*3) / 48000.0
	nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
	xRealtime := outDur / (nsPerOp / 1e9)
	b.ReportMetric(xRealtime, "x_realtime")
}

// BenchmarkStream_Chunk1000 benchmarks streaming with 1000-sample (62.5 ms) chunks.
func BenchmarkStream_Chunk1000(b *testing.B) { runStreamBench(b, 1000) }

// BenchmarkStream_Chunk4000 benchmarks streaming with 4000-sample (250 ms) chunks.
func BenchmarkStream_Chunk4000(b *testing.B) { runStreamBench(b, 4000) }

// BenchmarkStream_Chunk16000 benchmarks streaming with 16000-sample (1 s) chunks.
func BenchmarkStream_Chunk16000(b *testing.B) { runStreamBench(b, 16000) }
