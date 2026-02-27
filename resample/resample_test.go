package resample_test

import (
	"math"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/resample"
)

func TestNewFor_SameRate(t *testing.T) {
	r, err := resample.NewFor(48000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{0.1, 0.2, 0.3}
	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(in) {
		t.Fatalf("passthrough: got %d samples, want %d", len(out), len(in))
	}
}

func TestNewFor_InvalidRate(t *testing.T) {
	if _, err := resample.NewFor(0, 48000); err == nil {
		t.Fatal("expected error for inRate=0")
	}
	if _, err := resample.NewFor(48000, 0); err == nil {
		t.Fatal("expected error for outRate=0")
	}
}

func TestLinear_24kTo16k_Length(t *testing.T) {
	r, err := resample.NewFor(24000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	in := makeSine(440, 24000, 24000) // 1s @ 24kHz
	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	// Allow ±2 samples tolerance for rounding.
	if abs(len(out)-16000) > 2 {
		t.Fatalf("24k→16k: got %d samples, want ≈16000", len(out))
	}
}

func TestLinear_16kTo48k_Length(t *testing.T) {
	r, err := resample.NewFor(16000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	in := makeSine(440, 16000, 16000) // 1s @ 16kHz
	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	if abs(len(out)-48000) > 2 {
		t.Fatalf("16k→48k: got %d samples, want ≈48000", len(out))
	}
}

func TestLinear_Reset(t *testing.T) {
	r, err := resample.NewFor(24000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	in := makeSine(440, 24000, 4800)
	out1, _ := r.Process(in)
	r.Reset()
	out2, _ := r.Process(in)
	if len(out1) != len(out2) {
		t.Fatalf("after Reset, lengths differ: %d vs %d", len(out1), len(out2))
	}
}

func TestLinear_StreamingNoBoundaryJumps(t *testing.T) {
	r, err := resample.NewFor(24000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	// Feed a perfect 440 Hz sine in chunks, collect output.
	const totalIn = 48000 // 2s @ 24kHz
	const chunkSize = 4800
	in := makeSine(440, 24000, totalIn)

	var all []float32
	for i := 0; i < totalIn; i += chunkSize {
		chunk := in[i : i+chunkSize]
		out, err := r.Process(chunk)
		if err != nil {
			t.Fatal(err)
		}
		all = append(all, out...)
	}

	// Check no large discontinuities between consecutive output samples.
	maxJump := float32(0)
	for i := 1; i < len(all); i++ {
		if d := abs32(all[i] - all[i-1]); d > maxJump {
			maxJump = d
		}
	}
	// For a 440 Hz sine at 16kHz the max theoretical per-sample delta is
	// 2*pi*440/16000 ≈ 0.17. Allow 2× headroom.
	if maxJump > 0.35 {
		t.Fatalf("large boundary jump detected: max delta = %.4f", maxJump)
	}
}

// --- helpers ---

func makeSine(freq, sampleRate, numSamples int) []float32 {
	out := make([]float32, numSamples)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * float64(freq) * float64(i) / float64(sampleRate)))
	}
	return out
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
