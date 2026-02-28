package model_test

import (
	"os"
	"testing"

	"github.com/MeKo-Christian/flashsr-go/model"
)

func TestLoad_Embedded(t *testing.T) {
	t.Setenv("FLASHSR_MODEL_PATH", "")

	data, err := model.Load(model.Config{})
	if err != nil {
		t.Fatalf("Load with no path should use embedded model: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("embedded model is empty")
	}
}

func TestLoad_FromPath(t *testing.T) {
	data := []byte("fake model bytes")

	f, err := os.CreateTemp(t.TempDir(), "model*.onnx")
	if err != nil {
		t.Fatal(err)
	}

	_, err = f.Write(data)
	if err != nil {
		t.Fatal(err)
	}

	f.Close()

	got, err := model.Load(model.Config{Path: f.Name()})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if string(got) != string(data) {
		t.Fatalf("got %q, want %q", got, data)
	}
}

func TestLoad_EnvOverride(t *testing.T) {
	data := []byte("env model bytes")

	f, err := os.CreateTemp(t.TempDir(), "model*.onnx")
	if err != nil {
		t.Fatal(err)
	}

	_, err = f.Write(data)
	if err != nil {
		t.Fatal(err)
	}

	f.Close()

	t.Setenv("FLASHSR_MODEL_PATH", f.Name())
	// Even though Config.Path is empty, env should win.
	got, err := model.Load(model.Config{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if string(got) != string(data) {
		t.Fatalf("env override: got %q, want %q", got, data)
	}
}

func TestLoad_HashVerification_BadData(t *testing.T) {
	data := []byte("wrong model bytes")

	f, err := os.CreateTemp(t.TempDir(), "model*.onnx")
	if err != nil {
		t.Fatal(err)
	}

	_, err = f.Write(data)
	if err != nil {
		t.Fatal(err)
	}

	f.Close()

	_, err = model.Load(model.Config{Path: f.Name(), VerifyHash: true})
	if err == nil {
		t.Fatal("expected hash mismatch error")
	}
}
