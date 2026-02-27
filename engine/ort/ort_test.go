package ort_test

import (
	"os"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/engine"
	"github.com/MeKo-Christian/flashsr-go/engine/ort"
)

// Compile-time check: *ort.Engine satisfies engine.Engine.
var _ engine.Engine = (*ort.Engine)(nil)

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

// TestSmoke requires a real ORT library and the FlashSR model.
// It is skipped when FLASHSR_ORT_LIB is not set.
func TestSmoke_Run(t *testing.T) {
	libPath := os.Getenv("FLASHSR_ORT_LIB")
	if libPath == "" {
		t.Skip("FLASHSR_ORT_LIB not set; skipping ORT smoke test")
	}

	// Will be enabled in Phase 1 once session creation is wired.
	t.Skip("ORT session creation not yet implemented — Phase 1 TODO")
}
