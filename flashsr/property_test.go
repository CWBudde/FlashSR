package flashsr_test

import (
	"testing"

	"github.com/cwbudde/flashsr-go/flashsr"
	"github.com/cwbudde/flashsr-go/internal/testutil"
)

// TestProperty_NoNaN verifies batch output contains no NaN or Inf values.
func TestProperty_NoNaN(t *testing.T) {
	u := flashsr.NewWithEngine(&mockEngine{scale: 3})

	for _, sig := range propertySignals(t) {
		out, err := u.Upsample16kTo48k(sig.pcm)
		if err != nil {
			t.Fatalf("%s: Upsample16kTo48k: %v", sig.name, err)
		}

		if testutil.HasNaNOrInf(out) {
			t.Errorf("%s: output contains NaN or Inf", sig.name)
		}
	}
}

// TestProperty_PeakNormalized verifies output peak is ≤ 1.0 after upsampling.
func TestProperty_PeakNormalized(t *testing.T) {
	u := flashsr.NewWithEngine(&mockEngine{scale: 3})

	for _, sig := range propertySignals(t) {
		out, err := u.Upsample16kTo48k(sig.pcm)
		if err != nil {
			t.Fatalf("%s: Upsample16kTo48k: %v", sig.name, err)
		}

		if peak := testutil.PeakAbs(out); peak > 1.0 {
			t.Errorf("%s: peak %.6f > 1.0 (not normalised)", sig.name, peak)
		}
	}
}

// TestProperty_OutputRate verifies output length is exactly 3× input length.
func TestProperty_OutputRate(t *testing.T) {
	u := flashsr.NewWithEngine(&mockEngine{scale: 3})

	for _, sig := range propertySignals(t) {
		out, err := u.Upsample16kTo48k(sig.pcm)
		if err != nil {
			t.Fatalf("%s: Upsample16kTo48k: %v", sig.name, err)
		}

		want := len(sig.pcm) * 3
		if len(out) != want {
			t.Errorf("%s: output length %d, want %d (3×%d)", sig.name, len(out), want, len(sig.pcm))
		}
	}
}

// propertySignal bundles a named test signal for property-based iteration.
type propertySignal struct {
	name string
	pcm  []float32
}

func propertySignals(t *testing.T) []propertySignal {
	t.Helper()

	return []propertySignal{
		{"sine_440Hz", testutil.Sine(440, 16000, 16000)},
		{"sine_sweep", testutil.SineSweep(50, 4000, 16000, 16000)},
		{"pink_noise", testutil.PinkNoise(42, 16000)},
	}
}
