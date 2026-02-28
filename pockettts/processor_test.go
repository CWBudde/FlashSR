package pockettts_test

import (
	"math"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/engine"
	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/pockettts"
)

// mockEngine triples every sample (simulates 3× upsampling).
type mockEngine struct{}

func (e *mockEngine) Run(in []float32) ([]float32, error) {
	out := make([]float32, len(in)*3)
	for i, v := range in {
		out[i*3] = v
		out[i*3+1] = v
		out[i*3+2] = v
	}

	return out, nil
}
func (e *mockEngine) Close() error            { return nil }
func (e *mockEngine) Info() engine.EngineInfo { return engine.EngineInfo{Provider: "mock"} }

func mockUpsampler() *flashsr.Upsampler {
	return flashsr.NewWithEngine(&mockEngine{})
}

//nolint:unparam // Keep freq parameter for readability in tests.
func makeSine(freq, sampleRate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * float64(freq) * float64(i) / float64(sampleRate)))
	}

	return out
}

// --- NewFlashSRProcessor ---

func TestNewFlashSRProcessor_NilUpsampler(t *testing.T) {
	_, err := pockettts.NewFlashSRProcessor(nil, 24000)
	if err == nil {
		t.Fatal("expected error for nil upsampler")
	}
}

func TestNewFlashSRProcessor_InvalidRate(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	_, err := pockettts.NewFlashSRProcessor(u, 0)
	if err == nil {
		t.Fatal("expected error for rate=0")
	}

	_, err = pockettts.NewFlashSRProcessor(u, -1)
	if err == nil {
		t.Fatal("expected error for negative rate")
	}
}

func TestNewFlashSRProcessor_ValidRates(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	for _, rate := range []int{16000, 24000, 44100, 48000} {
		_, err := pockettts.NewFlashSRProcessor(u, rate)
		if err != nil {
			t.Errorf("rate %d: unexpected error: %v", rate, err)
		}
	}
}

// --- Process ---

func TestProcess_16kPassthrough(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, err := pockettts.NewFlashSRProcessor(u, 16000)
	if err != nil {
		t.Fatal(err)
	}

	in := makeSine(440, 16000, 16000) // 1 s

	out, rate, err := p.Process(in, 16000)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	if rate != 48000 {
		t.Errorf("output rate: got %d, want 48000", rate)
	}

	// mock engine is 3× so output length ≈ 3× input
	if len(out) < len(in)*2 || len(out) > len(in)*4 {
		t.Errorf("output length %d out of expected range [%d, %d]", len(out), len(in)*2, len(in)*4)
	}
}

func TestProcess_24kResampled(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, err := pockettts.NewFlashSRProcessor(u, 24000)
	if err != nil {
		t.Fatal(err)
	}

	in := makeSine(440, 24000, 24000) // 1 s @ 24 kHz

	out, rate, err := p.Process(in, 24000)
	if err != nil {
		t.Fatalf("Process (24k): %v", err)
	}

	if rate != 48000 {
		t.Errorf("output rate: got %d, want 48000", rate)
	}

	if len(out) == 0 {
		t.Error("output is empty")
	}
}

func TestProcess_44kResampled(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, err := pockettts.NewFlashSRProcessor(u, 44100)
	if err != nil {
		t.Fatal(err)
	}

	in := makeSine(440, 44100, 44100) // 1 s @ 44.1 kHz

	_, rate, err := p.Process(in, 44100)
	if err != nil {
		t.Fatalf("Process (44.1k): %v", err)
	}

	if rate != 48000 {
		t.Errorf("output rate: got %d, want 48000", rate)
	}
}

func TestProcess_EmptyInput(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, _ := pockettts.NewFlashSRProcessor(u, 16000)

	_, _, err := p.Process(nil, 16000)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestProcess_RateMismatch(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, _ := pockettts.NewFlashSRProcessor(u, 24000)

	_, _, err := p.Process(makeSine(440, 16000, 1000), 16000)
	if err == nil {
		t.Fatal("expected error for mismatched sample rate")
	}
}

func TestProcess_NoNaN(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, _ := pockettts.NewFlashSRProcessor(u, 24000)
	in := makeSine(440, 24000, 4800)

	out, _, err := p.Process(in, 24000)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("NaN/Inf at index %d", i)
		}
	}
}

// --- ProcessWAVResult ---

func TestProcessWAVResult(t *testing.T) {
	u := mockUpsampler()
	defer u.Close()

	p, _ := pockettts.NewFlashSRProcessor(u, 24000)

	r := pockettts.WAVResult{
		PCM:        makeSine(440, 24000, 4800),
		SampleRate: 24000,
	}

	out, err := pockettts.ProcessWAVResult(p, r)
	if err != nil {
		t.Fatalf("ProcessWAVResult: %v", err)
	}

	if len(out) == 0 {
		t.Error("output is empty")
	}
}

// --- Int16ToFloat32 ---

func TestInt16ToFloat32_Zero(t *testing.T) {
	out := pockettts.Int16ToFloat32([]int16{0, 0, 0})
	for i, v := range out {
		if v != 0 {
			t.Errorf("[%d]: got %f, want 0", i, v)
		}
	}
}

func TestInt16ToFloat32_MaxMin(t *testing.T) {
	out := pockettts.Int16ToFloat32([]int16{math.MaxInt16, math.MinInt16})

	if out[0] > 1.0 || out[0] < 0.99 {
		t.Errorf("MaxInt16 → %f, want ~1.0", out[0])
	}

	if out[1] < -1.0 || out[1] > -0.99 {
		t.Errorf("MinInt16 → %f, want ~-1.0", out[1])
	}
}

func TestInt16ToFloat32_Roundtrip(t *testing.T) {
	// Encode sine as int16, decode, check magnitude is within ±2 LSB.
	const n = 1600

	in := make([]int16, n)
	for i := range in {
		in[i] = int16(math.Sin(2*math.Pi*440*float64(i)/16000) * 32767)
	}

	out := pockettts.Int16ToFloat32(in)
	if len(out) != n {
		t.Fatalf("length: got %d, want %d", len(out), n)
	}

	const epsilon = 2.0 / 32768.0 // 2 LSB

	for i, v := range out {
		want := float32(in[i]) / 32768.0
		if diff := float32(math.Abs(float64(v - want))); diff > epsilon {
			t.Fatalf("[%d]: got %.6f, want %.6f (diff %.6f)", i, v, want, diff)
		}
	}
}
