// Package model handles loading the FlashSR ONNX model.
// Priority: FLASHSR_MODEL_PATH env → Config.Path → embedded binary.
package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"github.com/cwbudde/flashsr-go/assets"
)

// ExpectedSHA256 is the SHA256 hex digest of the pinned model artefact.
// Source: YatharthS/FlashSR @ onnx/model.onnx.
const ExpectedSHA256 = "e255c76b227f16f7f392cc43677c38bd2c5aa129f042a2ba3eb03fb29e470c7a"

// embeddedModel is the FlashSR ONNX model compiled into the binary via assets.
var embeddedModel = assets.ModelONNX

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
		err := verifyHash(data, ExpectedSHA256)
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

func verifyHash(data []byte, want string) error {
	sum := sha256.Sum256(data)

	got := hex.EncodeToString(sum[:])
	if got != want {
		return fmt.Errorf("model: hash mismatch: got %s, want %s", got, want)
	}

	return nil
}
