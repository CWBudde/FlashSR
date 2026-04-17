package ort_test

import (
	"os"
	"sync"
	"testing"

	"github.com/cwbudde/flashsr-go/engine"
	"github.com/cwbudde/flashsr-go/engine/ort"
)

// Compile-time check: *ort.Engine satisfies engine.Engine.
var _ engine.Engine = (*ort.Engine)(nil)

// ---- error-path tests (no ORT lib required) ----

func TestNew_NoLibPath(t *testing.T) {
	t.Setenv("FLASHSR_ORT_LIB", "")

	_, err := ort.New([]byte("fake"), ort.Config{})
	if err == nil {
		t.Fatal("expected error when no library path provided")
	}
}

func TestNew_EmptyModel(t *testing.T) {
	_, err := ort.New(nil, ort.Config{LibraryPath: "/fake/path.so"})
	if err == nil {
		t.Fatal("expected error for empty model bytes")
	}
}

// ---- helpers for ORT-backed tests ----

// ortLib returns the ORT library path, or skips the test if unset.
func ortLib(t *testing.T) string {
	t.Helper()

	lib := os.Getenv("FLASHSR_ORT_LIB")
	if lib == "" {
		t.Skip("FLASHSR_ORT_LIB not set; skipping ORT integration test")
	}

	return lib
}

// loadModel reads the model from FLASHSR_MODEL_PATH, or skips if unset.
func loadModel(t *testing.T) []byte {
	t.Helper()

	p := os.Getenv("FLASHSR_MODEL_PATH")
	if p == "" {
		t.Skip("FLASHSR_MODEL_PATH not set; skipping model-based test")
	}

	data, err := os.ReadFile(p)
	if err != nil {
		t.Skipf("FLASHSR_MODEL_PATH %q not readable: %v", p, err)
	}

	return data
}

// ---- integration tests (skipped without ORT lib + model) ----

func TestSmoke_Info(t *testing.T) {
	lib := ortLib(t)
	model := loadModel(t)

	e, err := ort.New(model, ort.Config{LibraryPath: lib})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer e.Close()

	info := e.Info()
	if info.OrtVersion == "" {
		t.Error("OrtVersion is empty")
	}

	if info.InputName == "" {
		t.Error("InputName is empty")
	}

	if info.OutputName == "" {
		t.Error("OutputName is empty")
	}

	if info.InputRank < 2 || info.InputRank > 3 {
		t.Errorf("unexpected InputRank %d (want 2 or 3)", info.InputRank)
	}

	t.Logf("ORT version: %s  input=%q rank=%d  output=%q",
		info.OrtVersion, info.InputName, info.InputRank, info.OutputName)
}

func TestSmoke_Run_Shape(t *testing.T) {
	lib := ortLib(t)
	model := loadModel(t)

	e, err := ort.New(model, ort.Config{LibraryPath: lib})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer e.Close()

	const inSamples = 4000
	input := make([]float32, inSamples)

	out, err := e.Run(input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(out) != inSamples*3 {
		t.Fatalf("output length = %d, want %d (3× input)", len(out), inSamples*3)
	}
}

func TestSmoke_Run_NoNaN(t *testing.T) {
	lib := ortLib(t)
	model := loadModel(t)

	e, err := ort.New(model, ort.Config{LibraryPath: lib})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer e.Close()

	input := makeSine(440, 16000, 4000)

	out, err := e.Run(input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	for i, v := range out {
		if v != v { // NaN check: NaN != NaN
			t.Fatalf("NaN at output index %d", i)
		}
	}
}

// TestConcurrentNew verifies the sync.Once guard handles concurrent New calls
// without panicking. All goroutines share the same lib path so the first wins.
func TestConcurrentNew(t *testing.T) {
	lib := ortLib(t)
	model := loadModel(t)

	const n = 5
	var wg sync.WaitGroup
	errs := make([]error, n)

	wg.Add(n)

	for i := range n {
		go func(idx int) {
			defer wg.Done()

			e, err := ort.New(model, ort.Config{LibraryPath: lib})
			errs[idx] = err

			if e != nil {
				e.Close()
			}
		}(i)
	}

	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: %v", i, err)
		}
	}
}

// --- helpers ---

func makeSine(freq, sampleRate, n int) []float32 {
	out := make([]float32, n)

	const pi = 3.14159265358979323846
	for i := range out {
		out[i] = float32(sinTaylor(2 * pi * float64(freq) * float64(i) / float64(sampleRate)))
	}

	return out
}

// sinTaylor approximates sin(x) via a 9th-order Taylor series; avoids math import.
func sinTaylor(x float64) float64 {
	const pi = 3.14159265358979323846
	const tau = 2 * pi
	// Reduce to [-π, π]
	x -= float64(int(x/tau)) * tau
	if x > pi {
		x -= tau
	} else if x < -pi {
		x += tau
	}

	x2 := x * x

	return x * (1 - x2*(1.0/6-x2*(1.0/120-x2*(1.0/5040-x2/362880))))
}
