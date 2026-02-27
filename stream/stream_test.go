package stream_test

import (
	"math"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/engine"
	"github.com/MeKo-Christian/flashsr-go/stream"
)

// scaleEngine triples every sample and repeats it 3× to simulate 3× upsampling.
type scaleEngine struct{}

func (e *scaleEngine) Run(input []float32) ([]float32, error) {
	out := make([]float32, len(input)*3)
	for i, v := range input {
		out[i*3] = v
		out[i*3+1] = v
		out[i*3+2] = v
	}

	return out, nil
}
func (e *scaleEngine) Close() error            { return nil }
func (e *scaleEngine) Info() engine.EngineInfo { return engine.EngineInfo{Provider: "mock-scale"} }

func newStreamer(t *testing.T, cfg stream.Config) *stream.Streamer {
	t.Helper()
	return stream.New(&scaleEngine{}, cfg)
}

func TestStreamer_Write_Read_Basic(t *testing.T) {
	s := newStreamer(t, stream.Config{ChunkSize: 4000})
	defer s.Reset()

	input := make([]float32, 4000)

	err := s.Write(input)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}

	// There should be some output buffered after a full chunk.
	if s.Buffered() == 0 {
		t.Fatal("expected buffered output after writing one full chunk")
	}
}

func TestStreamer_OutputNoNaN(t *testing.T) {
	s := newStreamer(t, stream.Config{ChunkSize: 4000})

	input := makeSine(440, 16000, 8000)

	err := s.Write(input)
	if err != nil {
		t.Fatal(err)
	}

	out := make([]float32, s.Buffered())
	n, _ := s.Read(out)
	out = out[:n]

	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("bad value at index %d: %v", i, v)
		}
	}
}

func TestStreamer_Reset_Deterministic(t *testing.T) {
	s := newStreamer(t, stream.Config{ChunkSize: 4000})

	input := makeSine(440, 16000, 4000)

	err := s.Write(input)
	if err != nil {
		t.Fatal(err)
	}

	buf1 := make([]float32, s.Buffered())
	n1, _ := s.Read(buf1)
	buf1 = buf1[:n1]

	s.Reset()

	if err = s.Write(input); err != nil {
		t.Fatal(err)
	}

	buf2 := make([]float32, s.Buffered())
	n2, _ := s.Read(buf2)
	buf2 = buf2[:n2]

	if n1 != n2 {
		t.Fatalf("after Reset lengths differ: %d vs %d", n1, n2)
	}

	for i := range buf1 {
		if buf1[i] != buf2[i] {
			t.Fatalf("output differs at index %d: %.6f vs %.6f", i, buf1[i], buf2[i])
		}
	}
}

func TestStreamer_Flush(t *testing.T) {
	s := newStreamer(t, stream.Config{ChunkSize: 4000})

	// Write a partial chunk (< ChunkSize).
	input := make([]float32, 2000)

	err := s.Write(input)
	if err != nil {
		t.Fatal(err)
	}
	// Nothing processed yet.
	if s.Buffered() != 0 {
		t.Fatalf("expected 0 buffered before flush, got %d", s.Buffered())
	}

	if err = s.Flush(); err != nil {
		t.Fatal(err)
	}

	if s.Buffered() == 0 {
		t.Fatal("expected output after Flush")
	}
}

func TestStreamer_CrossfadeSmooth(t *testing.T) {
	s := newStreamer(t, stream.Config{ChunkSize: 4000})

	// Feed 3 full chunks of a sine wave.
	input := makeSine(440, 16000, 12000)
	for i := 0; i < 12000; i += 4000 {
		err := s.Write(input[i : i+4000])
		if err != nil {
			t.Fatal(err)
		}
	}

	out := make([]float32, s.Buffered())
	n, _ := s.Read(out)
	out = out[:n]

	if len(out) == 0 {
		t.Fatal("no output produced")
	}

	// Check no large discontinuities between consecutive output samples.
	maxJump := float32(0)
	for i := 1; i < len(out); i++ {
		if d := abs32(out[i] - out[i-1]); d > maxJump {
			maxJump = d
		}
	}
	// With the nearest-neighbour mock each input sample becomes three identical
	// output samples, so within-chunk steps already reach ~0.17 for a 440 Hz sine.
	// The crossfade must not introduce jumps dramatically larger than within-chunk
	// steps. Threshold 1.5 catches catastrophic failures; the golden tests
	// (Phase 6) verify real smoothness with the actual ONNX model.
	if maxJump > 1.5 {
		t.Fatalf("large discontinuity at chunk boundary: max jump = %.4f", maxJump)
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

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}

	return x
}
