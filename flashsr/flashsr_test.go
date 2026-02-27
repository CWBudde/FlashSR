package flashsr_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/engine"
	"github.com/MeKo-Christian/flashsr-go/flashsr"
)

// mockEngine satisfies engine.Engine for unit testing without ORT.
type mockEngine struct {
	scale int // upsample ratio (output = input * scale)
}

func (m *mockEngine) Run(input []float32) ([]float32, error) {
	out := make([]float32, len(input)*m.scale)
	for i, v := range input {
		for j := range m.scale {
			out[i*m.scale+j] = v
		}
	}

	return out, nil
}

func (m *mockEngine) Close() error            { return nil }
func (m *mockEngine) Info() engine.EngineInfo { return engine.EngineInfo{Provider: "mock"} }

func requireUpsampler(t *testing.T) *flashsr.Upsampler {
	t.Helper()
	return flashsr.NewWithEngine(&mockEngine{scale: 3})
}

func TestUpsample_Shape(t *testing.T) {
	u := requireUpsampler(t)
	defer u.Close()

	input := make([]float32, 4000)

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(out) != 12000 {
		t.Fatalf("got %d samples, want 12000", len(out))
	}
}

func TestUpsample_EmptyInput(t *testing.T) {
	u := requireUpsampler(t)
	defer u.Close()

	_, err := u.Upsample16kTo48k(nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestUpsample_ClampInput(t *testing.T) {
	// mockEngine echos samples; after peak-norm the output should be ≤ 1.0.
	u := requireUpsampler(t)
	defer u.Close()

	input := []float32{-2.0, 0.5, 2.0} // values outside [-1,1]

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, v := range out {
		if v > 1.0 || v < -1.0 {
			t.Fatalf("output sample %d = %.4f is outside [-1,1]", i, v)
		}
	}
}

func TestUpsample_NoNaN(t *testing.T) {
	u := requireUpsampler(t)
	defer u.Close()

	input := makeSine(440, 16000, 4000)

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, v := range out {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}

		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestUpsample_PeakNormalized(t *testing.T) {
	u := requireUpsampler(t)
	defer u.Close()

	input := makeSine(440, 16000, 4000)

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	peak := peakAbs(out)
	if peak > 1.0 {
		t.Fatalf("peak %.4f exceeds 1.0", peak)
	}
}

func TestNew_EngineInitError(t *testing.T) {
	// No model path + no ORT lib → both ErrModelLoad or ErrEngineInit are acceptable.
	t.Setenv("FLASHSR_MODEL_PATH", "")
	t.Setenv("FLASHSR_ORT_LIB", "")

	_, err := flashsr.New(flashsr.Config{})
	if err == nil {
		t.Fatal("expected error when no model/ORT available")
	}

	if !errors.Is(err, flashsr.ErrModelLoad) && !errors.Is(err, flashsr.ErrEngineInit) {
		t.Fatalf("unexpected error type: %v", err)
	}
}

// --- helpers ---

func makeSine(freq, sampleRate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * float64(freq) * float64(i) / float64(sampleRate)))
	}

	return out
}

func peakAbs(x []float32) float32 {
	peak := float32(0)
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > peak {
			peak = a
		}
	}

	return peak
}
