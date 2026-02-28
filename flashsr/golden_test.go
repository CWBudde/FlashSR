//go:build golden

package flashsr_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/internal/testutil"
)

const fixturesDir = "../internal/testutil/fixtures"

// requireOrtUpsampler creates a real Upsampler backed by ORT.
// Skips the test when FLASHSR_ORT_LIB is not set.
func requireOrtUpsampler(t *testing.T) *flashsr.Upsampler {
	t.Helper()

	lib := os.Getenv("FLASHSR_ORT_LIB")
	if lib == "" {
		t.Skip("FLASHSR_ORT_LIB not set; skipping golden test")
	}

	u, err := flashsr.New(flashsr.Config{ORTLibPath: lib})
	if err != nil {
		t.Fatalf("flashsr.New: %v", err)
	}

	t.Cleanup(func() { _ = u.Close() })

	return u
}

// requireFixture loads a numpy float32 fixture, skipping when the file is absent.
func requireFixture(t *testing.T, name string) []float32 {
	t.Helper()

	path := filepath.Join(fixturesDir, name)

	data, err := testutil.LoadNPYFloat32(path)
	if os.IsNotExist(err) {
		t.Skipf("fixture %q not found — run scripts/gen_fixtures.py to generate", name)
	}

	if err != nil {
		t.Fatalf("load fixture %q: %v", name, err)
	}

	return data
}

func TestGolden_Batch_Sine(t *testing.T) {
	u := requireOrtUpsampler(t)
	ref := requireFixture(t, "ref_batch_sine.npy")

	input := testutil.Sine(440, 16000, 16000)

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("Upsample16kTo48k: %v", err)
	}

	assertGolden(t, out, ref, "batch_sine")
}

func TestGolden_Batch_PinkNoise(t *testing.T) {
	u := requireOrtUpsampler(t)
	ref := requireFixture(t, "ref_batch_pink.npy")

	input := testutil.PinkNoise(42, 16000)

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("Upsample16kTo48k: %v", err)
	}

	assertGolden(t, out, ref, "batch_pink")
}

func TestGolden_Batch_SineSweep(t *testing.T) {
	u := requireOrtUpsampler(t)
	ref := requireFixture(t, "ref_batch_sweep.npy")

	input := testutil.SineSweep(50, 4000, 16000, 16000)

	out, err := u.Upsample16kTo48k(input)
	if err != nil {
		t.Fatalf("Upsample16kTo48k: %v", err)
	}

	assertGolden(t, out, ref, "batch_sweep")
}

// assertGolden checks RMS error ≤ -40 dB and that output is not clipped.
func assertGolden(t *testing.T, got, ref []float32, label string) {
	t.Helper()

	// Trim or pad to the shorter length for comparison.
	n := len(got)
	if len(ref) < n {
		n = len(ref)
	}

	rmsDB, err := testutil.RMSError(got[:n], ref[:n])
	if err != nil {
		t.Fatalf("%s: RMSError: %v", label, err)
	}

	t.Logf("%s: RMS error = %.2f dB (vs Python reference)", label, rmsDB)

	if rmsDB > -40.0 {
		t.Errorf("%s: RMS error %.2f dB > -40 dB threshold", label, rmsDB)
	}

	if peak := testutil.PeakAbs(got); peak > 1.0 {
		t.Errorf("%s: output clipped: peak = %.4f", label, peak)
	}
}
