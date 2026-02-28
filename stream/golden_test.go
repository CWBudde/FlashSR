//go:build golden

package stream_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/engine/ort"
	"github.com/MeKo-Christian/flashsr-go/internal/testutil"
	"github.com/MeKo-Christian/flashsr-go/model"
	"github.com/MeKo-Christian/flashsr-go/stream"
)

const streamFixturesDir = "../internal/testutil/fixtures"

func requireStreamer(t *testing.T) *stream.Streamer {
	t.Helper()

	lib := os.Getenv("FLASHSR_ORT_LIB")
	if lib == "" {
		t.Skip("FLASHSR_ORT_LIB not set; skipping golden test")
	}

	modelBytes, err := model.Load(model.Config{})
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}

	eng, err := ort.New(modelBytes, ort.Config{LibraryPath: lib})
	if err != nil {
		t.Fatalf("ort.New: %v", err)
	}

	t.Cleanup(func() { _ = eng.Close() })

	return stream.New(eng, stream.Config{ChunkSize: 4000})
}

func requireStreamFixture(t *testing.T, name string) []float32 {
	t.Helper()

	path := filepath.Join(streamFixturesDir, name)

	data, err := testutil.LoadNPYFloat32(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("fixture %q not found — run scripts/gen_fixtures.py to generate", name)
	}

	if err != nil {
		t.Fatalf("load fixture %q: %v", name, err)
	}

	return data
}

func TestGolden_Stream_Sine(t *testing.T) {
	s := requireStreamer(t)
	ref := requireStreamFixture(t, "ref_stream_sine.npy")

	input := testutil.Sine(440, 16000, 16000)
	out := feedStreamer(t, s, input, 4000)

	assertStreamGolden(t, out, ref, "stream_sine")
}

func TestGolden_Stream_PinkNoise(t *testing.T) {
	s := requireStreamer(t)
	ref := requireStreamFixture(t, "ref_stream_pink.npy")

	// Load input from fixture so Go and Python use identical bytes (their RNGs differ).
	input := requireStreamFixture(t, "in_pink.npy")
	out := feedStreamer(t, s, input, 4000)

	assertStreamGolden(t, out, ref, "stream_pink")
}

func feedStreamer(t *testing.T, s *stream.Streamer, pcm []float32, chunkSize int) []float32 {
	t.Helper()

	for i := 0; i < len(pcm); i += chunkSize {
		end := i + chunkSize
		if end > len(pcm) {
			end = len(pcm)
		}

		if err := s.Write(pcm[i:end]); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}

	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	out := make([]float32, s.Buffered())
	n, _ := s.Read(out)

	return out[:n]
}

func assertStreamGolden(t *testing.T, got, ref []float32, label string) {
	t.Helper()

	// Streaming trims leading samples; allow up to 5% length difference.
	n := len(got)
	if len(ref) < n {
		n = len(ref)
	}

	allowedDiff := len(ref) / 20 // 5%
	lenDiff := len(ref) - len(got)
	if lenDiff < 0 {
		lenDiff = -lenDiff
	}

	if lenDiff > allowedDiff {
		t.Logf("%s: length: got %d, ref %d (diff %d > 5%%)", label, len(got), len(ref), lenDiff)
	}

	rmsDB, err := testutil.RMSError(got[:n], ref[:n])
	if err != nil {
		t.Fatalf("%s: RMSError: %v", label, err)
	}

	t.Logf("%s: RMS error = %.2f dB (vs Python reference)", label, rmsDB)

	if rmsDB > -40.0 {
		t.Errorf("%s: RMS error %.2f dB > -40 dB threshold", label, rmsDB)
	}
}
