// Package flashsr is the public API for FlashSR audio super-resolution.
// It converts PCM float32 at 16 kHz to 48 kHz using a small ONNX model.
package flashsr

import (
	"errors"
	"fmt"
	"math"

	"github.com/MeKo-Christian/flashsr-go/engine"
	"github.com/MeKo-Christian/flashsr-go/engine/ort"
	"github.com/MeKo-Christian/flashsr-go/model"
	"github.com/MeKo-Christian/flashsr-go/resample"
)

// Config controls how the Upsampler is initialized.
type Config struct {
	// ModelPath overrides the embedded model. Falls back to FLASHSR_MODEL_PATH env.
	ModelPath string

	// ORTLibPath is the path to libonnxruntime. Falls back to FLASHSR_ORT_LIB env.
	ORTLibPath string

	// NumThreadsIntra sets ORT intra-op thread count (default: 1).
	NumThreadsIntra int

	// NumThreadsInter sets ORT inter-op thread count (default: 1).
	NumThreadsInter int

	// VerifyModelHash validates the model SHA256 against the pinned digest.
	VerifyModelHash bool

	// InputRate is the sample rate of audio passed to Upsample16kTo48k.
	// When non-zero and not 16000, audio is resampled to 16 kHz before inference.
	// Zero means "assume 16 kHz input" (no resampling).
	InputRate int

	// ResampleQuality controls the polyphase FIR quality used when resampling.
	// Ignored when InputRate == 0 or InputRate == 16000.
	ResampleQuality resample.Quality

	// InputName overrides the auto-detected model input tensor name.
	InputName string

	// OutputName overrides the auto-detected model output tensor name.
	OutputName string
}

// Upsampler performs audio super-resolution via the configured engine.
type Upsampler struct {
	eng       engine.Engine
	resampler resample.Resampler // non-nil when InputRate != 16000
}

// New creates an Upsampler. It loads the model, initializes the ORT session,
// and introspects the model's tensor layout. Call Close when done.
func New(cfg Config) (*Upsampler, error) {
	modelBytes, err := model.Load(model.Config{
		Path:       cfg.ModelPath,
		VerifyHash: cfg.VerifyModelHash,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrModelLoad, err)
	}

	eng, err := ort.New(modelBytes, ort.Config{
		LibraryPath:     cfg.ORTLibPath,
		NumThreadsIntra: cfg.NumThreadsIntra,
		NumThreadsInter: cfg.NumThreadsInter,
		InputName:       cfg.InputName,
		OutputName:      cfg.OutputName,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrEngineInit, err)
	}

	u := &Upsampler{eng: eng}

	if cfg.InputRate != 0 && cfg.InputRate != 16000 {
		rs, err := resample.NewPolyphase(cfg.InputRate, 16000,
			resample.WithQuality(cfg.ResampleQuality))
		if err != nil {
			_ = eng.Close()
			return nil, fmt.Errorf("%w: build resampler: %w", ErrEngineInit, err)
		}

		u.resampler = rs
	}

	return u, nil
}

// NewWithEngine creates an Upsampler from a pre-built Engine.
// Useful for testing with a mock engine.
func NewWithEngine(eng engine.Engine) *Upsampler {
	return &Upsampler{eng: eng}
}

// Upsample16kTo48k converts PCM float32 to 48 kHz.
// When the Upsampler was configured with a non-16kHz InputRate, the audio is
// resampled to 16 kHz before inference.
// Input must be in [-1, 1]. Output is peak-normalized to ≤ 0.999.
func (u *Upsampler) Upsample16kTo48k(x []float32) ([]float32, error) {
	if len(x) == 0 {
		return nil, errors.New("flashsr: input is empty")
	}

	if u.resampler != nil {
		var err error

		x, err = u.resampler.Process(x)
		if err != nil {
			return nil, fmt.Errorf("%w: resample to 16kHz: %w", ErrInferFailed, err)
		}

		if len(x) == 0 {
			return nil, errors.New("flashsr: resampler produced empty output")
		}
	}

	clamped := clamp(x)

	out, err := u.eng.Run(clamped)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInferFailed, err)
	}

	return peakNormalize(out, 0.999), nil
}

// EngineInfo returns metadata about the underlying engine and model.
func (u *Upsampler) EngineInfo() engine.EngineInfo {
	return u.eng.Info()
}

// Engine returns the underlying inference engine.
// Useful for passing to stream.New in streaming mode.
func (u *Upsampler) Engine() engine.Engine {
	return u.eng
}

// Close releases the engine session resources.
func (u *Upsampler) Close() error {
	err := u.eng.Close()
	if err != nil {
		return fmt.Errorf("flashsr: close engine: %w", err)
	}

	return nil
}

// clamp returns a copy of x with every sample clamped to [-1, 1].
func clamp(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}

		out[i] = v
	}

	return out
}

// peakNormalize scales x so that its absolute peak equals target.
// If peak is zero (silence), x is returned unchanged.
func peakNormalize(x []float32, target float32) []float32 {
	peak := float32(0)
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > peak {
			peak = a
		}
	}

	if peak == 0 {
		return x
	}

	scale := target / peak

	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * scale
	}

	return out
}
