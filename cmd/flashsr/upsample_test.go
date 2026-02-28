package main

import (
	"math"
	"os"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/engine"
	"github.com/MeKo-Christian/flashsr-go/flashsr"
)

// TestWAVRoundtrip verifies writeWAV → readWAV preserves sample rate and
// that decoded samples are close to the originals (16-bit quantisation noise).
func TestWAVRoundtrip(t *testing.T) {
	const (
		sr      = 16000
		n       = 1600 // 0.1 s
		freq    = 440
		epsilon = float64(2.0 / 65535) // 1 LSB at 16-bit
	)

	input := makeSineWAV(freq, sr, n)

	tmp := t.TempDir() + "/roundtrip.wav"

	if err := writeWAV(tmp, input, sr); err != nil {
		t.Fatalf("writeWAV: %v", err)
	}

	got, gotSR, err := readWAV(tmp)
	if err != nil {
		t.Fatalf("readWAV: %v", err)
	}

	if gotSR != sr {
		t.Errorf("sample rate: got %d, want %d", gotSR, sr)
	}

	if len(got) != len(input) {
		t.Fatalf("length: got %d, want %d", len(got), len(input))
	}

	for i := range input {
		diff := math.Abs(float64(got[i]) - float64(input[i]))
		if diff > epsilon {
			t.Fatalf("sample[%d]: got %.6f, want %.6f (diff %.6f > epsilon %.6f)",
				i, got[i], input[i], diff, epsilon)
		}
	}
}

// TestUpsample_44kInputViaMock verifies that a 44.1kHz WAV flows through the
// pipeline without being rejected (non-16kHz is now resampled, not refused).
// Uses a mock engine — no ORT required.
func TestUpsample_44kInputViaMock(t *testing.T) {
	const sr = 44100

	dir := t.TempDir()
	inPath := dir + "/in44k.wav"
	outPath := dir + "/out.wav"

	if err := writeWAV(inPath, makeSineWAV(440, sr, sr/10), sr); err != nil {
		t.Fatalf("write 44.1kHz WAV: %v", err)
	}

	u := flashsr.NewWithEngine(&mockScaleEngine{})

	// runUpsampleWithUpsampler reads the WAV and runs inference regardless of
	// the WAV's sample rate; the mock 3×-scales whatever it receives.
	if err := runUpsampleWithUpsampler(u, upsampleFlags{
		input:  inPath,
		output: outPath,
	}); err != nil {
		t.Fatalf("runUpsampleWithUpsampler (44.1kHz): %v", err)
	}

	_, gotSR, err := readWAV(outPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	if gotSR != 48000 {
		t.Errorf("output sample rate: got %d, want 48000", gotSR)
	}
}

// TestUpsample_MockEngine uses a mock engine to test WAV I/O without ORT.
func TestUpsample_MockEngine(t *testing.T) {
	const (
		sr   = 16000
		n    = 4000
		freq = 440
	)

	dir := t.TempDir()
	inPath := dir + "/in.wav"
	outPath := dir + "/out.wav"

	if err := writeWAV(inPath, makeSineWAV(freq, sr, n), sr); err != nil {
		t.Fatalf("write input WAV: %v", err)
	}

	u := flashsr.NewWithEngine(&mockScaleEngine{})

	if err := runUpsampleWithUpsampler(u, upsampleFlags{
		input:  inPath,
		output: outPath,
	}); err != nil {
		t.Fatalf("runUpsampleWithUpsampler: %v", err)
	}

	samples, gotSR, err := readWAV(outPath)
	if err != nil {
		t.Fatalf("read output WAV: %v", err)
	}

	if gotSR != 48000 {
		t.Errorf("output sample rate: got %d, want 48000", gotSR)
	}

	if len(samples) == 0 {
		t.Error("output has no samples")
	}
}

// TestUpsample_24kInput verifies that a 24kHz WAV is resampled to 16kHz before
// inference (end-to-end, mock engine, no ORT required).
func TestUpsample_24kInput(t *testing.T) {
	const (
		sr   = 24000
		n    = 4800 // 0.2 s @ 24kHz
		freq = 440
	)

	dir := t.TempDir()
	inPath := dir + "/in24k.wav"
	outPath := dir + "/out48k.wav"

	if err := writeWAV(inPath, makeSineWAV(freq, sr, n), sr); err != nil {
		t.Fatalf("write 24kHz WAV: %v", err)
	}

	u := flashsr.NewWithEngine(&mockScaleEngine{})

	if err := runUpsampleWithUpsampler(u, upsampleFlags{
		input:  inPath,
		output: outPath,
	}); err != nil {
		t.Fatalf("runUpsampleWithUpsampler (24kHz): %v", err)
	}

	_, gotSR, err := readWAV(outPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	if gotSR != 48000 {
		t.Errorf("output sample rate: got %d, want 48000", gotSR)
	}
}

// TestCLI_Upsample_Basic runs a full upsample through the CLI pipeline.
// Skips when FLASHSR_ORT_LIB is not set.
func TestCLI_Upsample_Basic(t *testing.T) {
	ortLib := os.Getenv("FLASHSR_ORT_LIB")
	if ortLib == "" {
		t.Skip("FLASHSR_ORT_LIB not set; skipping ORT integration test")
	}

	const (
		sr   = 16000
		n    = sr / 2 // 0.5 s
		freq = 440
	)

	dir := t.TempDir()
	inPath := dir + "/in.wav"
	outPath := dir + "/out.wav"

	if err := writeWAV(inPath, makeSineWAV(freq, sr, n), sr); err != nil {
		t.Fatalf("write input WAV: %v", err)
	}

	if err := runUpsample(upsampleFlags{
		input:  inPath,
		output: outPath,
		ortLib: ortLib,
	}); err != nil {
		t.Fatalf("runUpsample: %v", err)
	}

	samples, gotSR, err := readWAV(outPath)
	if err != nil {
		t.Fatalf("read output WAV: %v", err)
	}

	if gotSR != 48000 {
		t.Errorf("output sample rate: got %d, want 48000", gotSR)
	}

	// Output should be approximately 3× the input length.
	wantMin := n * 2
	wantMax := n * 4
	if len(samples) < wantMin || len(samples) > wantMax {
		t.Errorf("output length %d outside expected range [%d, %d]", len(samples), wantMin, wantMax)
	}
}

// TestUpsample_Streaming runs the streaming path end-to-end.
// Skips when FLASHSR_ORT_LIB is not set.
func TestUpsample_Streaming(t *testing.T) {
	ortLib := os.Getenv("FLASHSR_ORT_LIB")
	if ortLib == "" {
		t.Skip("FLASHSR_ORT_LIB not set; skipping ORT integration test")
	}

	const (
		sr   = 16000
		n    = sr // 1.0 s
		freq = 440
	)

	dir := t.TempDir()
	inPath := dir + "/in.wav"
	outPath := dir + "/out.wav"

	if err := writeWAV(inPath, makeSineWAV(freq, sr, n), sr); err != nil {
		t.Fatalf("write input WAV: %v", err)
	}

	if err := runUpsample(upsampleFlags{
		input:     inPath,
		output:    outPath,
		ortLib:    ortLib,
		streaming: true,
		chunkSize: 4000,
	}); err != nil {
		t.Fatalf("runUpsample (streaming): %v", err)
	}

	_, gotSR, err := readWAV(outPath)
	if err != nil {
		t.Fatalf("read output WAV: %v", err)
	}

	if gotSR != 48000 {
		t.Errorf("output sample rate: got %d, want 48000", gotSR)
	}
}

// mockScaleEngine triples every sample (3× upsampling mock).
type mockScaleEngine struct{}

func (e *mockScaleEngine) Run(in []float32) ([]float32, error) {
	out := make([]float32, len(in)*3)
	for i, v := range in {
		out[i*3] = v
		out[i*3+1] = v
		out[i*3+2] = v
	}

	return out, nil
}

func (e *mockScaleEngine) Close() error            { return nil }
func (e *mockScaleEngine) Info() engine.EngineInfo { return engine.EngineInfo{Provider: "mock"} }

// --- helpers ---

func makeSineWAV(freq, sampleRate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * float64(freq) * float64(i) / float64(sampleRate)))
	}

	return out
}
