// Package model handles loading the FlashSR ONNX model.
// Priority: FLASHSR_MODEL_PATH env → Config.Path → embedded binary.
package model

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
)

// ExpectedSHA256 is the SHA256 hex digest of the pinned model artefact.
// It will be populated once the model is downloaded in Phase 2.
const ExpectedSHA256 = ""

// embeddedModel holds the model bytes compiled into the binary.
// Populated via go:embed once assets/model.onnx is present.
//
// TODO(phase2): uncomment after downloading assets/model.onnx
//
//	//go:embed ../assets/model.onnx
//	var embeddedModel []byte
var embeddedModel []byte

// Config controls how the model is loaded.
type Config struct {
	// Path overrides the embedded model with a file on disk.
	Path string

	// VerifyHash checks the SHA256 of loaded bytes against ExpectedSHA256.
	// Ignored when ExpectedSHA256 is empty.
	VerifyHash bool
}

// Load returns the model bytes according to the priority order:
//  1. FLASHSR_MODEL_PATH environment variable
//  2. Config.Path
//  3. Embedded binary (assets/model.onnx)
func Load(cfg Config) ([]byte, error) {
	path := os.Getenv("FLASHSR_MODEL_PATH")
	if path == "" {
		path = cfg.Path
	}

	var data []byte
	var err error

	if path != "" {
		data, err = os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("model: read %q: %w", path, err)
		}
	} else {
		if len(embeddedModel) == 0 {
			return nil, errors.New("model: no embedded model — set FLASHSR_MODEL_PATH or Config.Path, or embed assets/model.onnx")
		}
		data = embeddedModel
	}

	if cfg.VerifyHash && ExpectedSHA256 != "" {
		if err := verifyHash(data, ExpectedSHA256); err != nil {
			return nil, err
		}
	}

	return data, nil
}

func verifyHash(data []byte, want string) error {
	sum := sha256.Sum256(data)
	got := fmt.Sprintf("%x", sum[:])
	if got != want {
		return fmt.Errorf("model: hash mismatch: got %s, want %s", got, want)
	}
	return nil
}
