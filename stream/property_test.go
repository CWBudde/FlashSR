package stream_test

import (
	"testing"

	"github.com/MeKo-Christian/flashsr-go/internal/testutil"
	"github.com/MeKo-Christian/flashsr-go/stream"
)

// TestProperty_Stream_NoNaN verifies streaming output has no NaN or Inf.
func TestProperty_Stream_NoNaN(t *testing.T) {
	for _, sig := range streamPropertySignals(t) {
		s := stream.New(&scaleEngine{}, stream.Config{ChunkSize: 4000})

		if err := s.Write(sig.pcm); err != nil {
			t.Fatalf("%s: Write: %v", sig.name, err)
		}

		if err := s.Flush(); err != nil {
			t.Fatalf("%s: Flush: %v", sig.name, err)
		}

		out := make([]float32, s.Buffered())
		n, _ := s.Read(out)
		out = out[:n]

		if testutil.HasNaNOrInf(out) {
			t.Errorf("%s: streaming output contains NaN or Inf", sig.name)
		}
	}
}

// TestProperty_Stream_OutputRate verifies streaming output length is roughly 3×
// the input. The Streamer trims the first chunk and uses overlap, so we allow a
// generous window of [1×, 4×] rather than exactly 3×.
func TestProperty_Stream_OutputRate(t *testing.T) {
	const chunkSize = 4000

	for _, sig := range streamPropertySignals(t) {
		s := stream.New(&scaleEngine{}, stream.Config{ChunkSize: chunkSize})

		if err := s.Write(sig.pcm); err != nil {
			t.Fatalf("%s: Write: %v", sig.name, err)
		}

		if err := s.Flush(); err != nil {
			t.Fatalf("%s: Flush: %v", sig.name, err)
		}

		out := make([]float32, s.Buffered())
		n, _ := s.Read(out)

		lo := len(sig.pcm)           // at least 1× (trims happen at start)
		hi := len(sig.pcm)*3 + 6000  // at most 3× + small buffer for overlap/flush
		if n < lo || n > hi {
			t.Errorf("%s: output length %d outside [%d, %d]", sig.name, n, lo, hi)
		}
	}
}

type streamPropertySignal struct {
	name string
	pcm  []float32
}

func streamPropertySignals(t *testing.T) []streamPropertySignal {
	t.Helper()

	return []streamPropertySignal{
		{"sine_440Hz", testutil.Sine(440, 16000, 16000)},
		{"sine_sweep", testutil.SineSweep(50, 4000, 16000, 16000)},
		{"pink_noise", testutil.PinkNoise(42, 16000)},
	}
}
