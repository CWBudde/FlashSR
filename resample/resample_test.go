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
	_, err := resample.NewFor(0, 48000)
	if err == nil {
		t.Fatal("expected error for inRate=0")
	}

	_, err = resample.NewFor(48000, 0)
	if err == nil {
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

// --- polyphase tests ---

func TestPolyphase_24kTo16k_Length(t *testing.T) {
	r, err := resample.NewPolyphase(24000, 16000)
	if err != nil {
		t.Fatal(err)
	}

	in := makeSine(440, 24000, 24000) // 1s @ 24kHz

	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}

	if abs(len(out)-16000) > 5 {
		t.Fatalf("24k→16k polyphase: got %d samples, want ≈16000", len(out))
	}
}

func TestPolyphase_16kTo48k_Length(t *testing.T) {
	r, err := resample.NewPolyphase(16000, 48000)
	if err != nil {
		t.Fatal(err)
	}

	in := makeSine(440, 16000, 16000)

	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}

	if abs(len(out)-48000) > 5 {
		t.Fatalf("16k→48k polyphase: got %d samples, want ≈48000", len(out))
	}
}

func TestPolyphase_QualityModes(t *testing.T) {
	rates := [][2]int{{24000, 16000}, {44100, 16000}, {8000, 16000}}
	qualities := []resample.Quality{resample.QualityFast, resample.QualityBalanced, resample.QualityBest}

	for _, rr := range rates {
		for _, q := range qualities {
			r, err := resample.NewPolyphase(rr[0], rr[1], resample.WithQuality(q))
			if err != nil {
				t.Fatalf("%d→%d quality=%d: %v", rr[0], rr[1], q, err)
			}

			in := makeSine(440, rr[0], rr[0])

			out, err := r.Process(in)
			if err != nil {
				t.Fatalf("%d→%d quality=%d: Process: %v", rr[0], rr[1], q, err)
			}

			if len(out) == 0 {
				t.Errorf("%d→%d quality=%d: empty output", rr[0], rr[1], q)
			}
		}
	}
}

func TestPolyphase_Reset(t *testing.T) {
	r, err := resample.NewPolyphase(24000, 16000)
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

	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("after Reset, sample[%d] differs: %.6f vs %.6f", i, out1[i], out2[i])
		}
	}
}

func TestPolyphase_SameRate_Passthrough(t *testing.T) {
	r, err := resample.NewPolyphase(16000, 16000)
	if err != nil {
		t.Fatal(err)
	}

	in := makeSine(440, 16000, 1600)

	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != len(in) {
		t.Fatalf("passthrough: got %d, want %d", len(out), len(in))
	}
}

func TestPolyphase_NoNaN(t *testing.T) {
	for _, q := range []resample.Quality{resample.QualityFast, resample.QualityBalanced, resample.QualityBest} {
		r, _ := resample.NewPolyphase(44100, 16000, resample.WithQuality(q))
		in := makeSine(440, 44100, 44100)
		out, _ := r.Process(in)

		for i, v := range out {
			if v != v { // NaN check
				t.Fatalf("quality=%d: NaN at index %d", q, i)
			}
		}
	}
}

// --- helpers ---

//nolint:unparam // Keep freq parameter for readability in tests.
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
