// Package resample provides sample-rate conversion for FlashSR pipelines.
//
// Default build: stateful linear interpolating resampler (zero external dependencies).
// Build with -tags algodsp to swap in github.com/cwbudde/algo-dsp/dsp/resample
// (polyphase FIR, quality modes) for higher output quality.
package resample

import (
	"errors"
	"fmt"
)

// ErrInvalidRate is returned when inRate or outRate is ≤ 0.
var ErrInvalidRate = errors.New("resample: sample rates must be positive")

// Resampler converts PCM float32 between sample rates in a streaming-friendly manner.
type Resampler interface {
	// Process converts a chunk of input samples.
	// Returns converted samples; length ≈ len(in) * outRate / inRate.
	Process(in []float32) ([]float32, error)

	// Reset clears stateful history so the resampler can be reused for a new stream.
	Reset()
}

// NewFor returns a Resampler for the given rate pair.
// When inRate == outRate it returns a pass-through resampler.
func NewFor(inRate, outRate int) (Resampler, error) {
	if inRate <= 0 || outRate <= 0 {
		return nil, fmt.Errorf("%w: in=%d out=%d", ErrInvalidRate, inRate, outRate)
	}

	if inRate == outRate {
		return &passthrough{}, nil
	}

	return newResampler(inRate, outRate)
}

// passthrough is a no-op resampler for inRate == outRate.
type passthrough struct{}

func (p *passthrough) Process(in []float32) ([]float32, error) {
	out := make([]float32, len(in))
	copy(out, in)

	return out, nil
}
func (p *passthrough) Reset() {}
