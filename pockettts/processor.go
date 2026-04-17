// Package pockettts integrates FlashSR with go-call-pocket-tts output.
// It converts Pocket-TTS WAV results (typically 24 kHz, int16 PCM) into
// 48 kHz float32 PCM using FlashSR super-resolution.
package pockettts

import (
	"errors"
	"fmt"

	"github.com/cwbudde/flashsr-go/flashsr"
	"github.com/cwbudde/flashsr-go/resample"
)

// WAVResult mirrors the WAV output produced by go-call-pocket-tts to avoid
// a hard dependency on that module.
type WAVResult struct {
	// PCM holds float32 samples in [-1, 1] at SampleRate Hz.
	PCM []float32
	// SampleRate is the sample rate of PCM (typically 24000).
	SampleRate int
}

// PostProcessor takes PCM audio at an arbitrary input sample rate and returns
// 48 kHz super-resolved PCM.
type PostProcessor interface {
	Process(pcm []float32, inSampleRate int) (out []float32, outSampleRate int, err error)
}

// ProcessWAVResult is a convenience wrapper that passes r.PCM and r.SampleRate
// to p.Process.
func ProcessWAVResult(p PostProcessor, r WAVResult) ([]float32, error) {
	out, _, err := p.Process(r.PCM, r.SampleRate)
	if err != nil {
		return nil, fmt.Errorf("pockettts: post-process WAV result: %w", err)
	}

	return out, nil
}

// FlashSRProcessor implements PostProcessor using a FlashSR Upsampler.
// Audio is resampled to 16 kHz when inSampleRate != 16000, then upscaled to 48 kHz.
type FlashSRProcessor struct {
	upsampler *flashsr.Upsampler
	resampler resample.Resampler // nil when inSampleRate == 16000
	inRate    int
}

// NewFlashSRProcessor creates a FlashSRProcessor for the given input sample rate.
// When inputSampleRate == 16000 no pre-resampling is performed.
// When inputSampleRate != 16000, a polyphase FIR resampler is built automatically.
func NewFlashSRProcessor(u *flashsr.Upsampler, inputSampleRate int) (*FlashSRProcessor, error) {
	if u == nil {
		return nil, errors.New("pockettts: upsampler must not be nil")
	}

	if inputSampleRate <= 0 {
		return nil, fmt.Errorf("pockettts: invalid input sample rate %d", inputSampleRate)
	}

	p := &FlashSRProcessor{upsampler: u, inRate: inputSampleRate}

	if inputSampleRate != 16000 {
		rs, err := resample.NewPolyphase(inputSampleRate, 16000)
		if err != nil {
			return nil, fmt.Errorf("pockettts: build resampler %d→16000: %w", inputSampleRate, err)
		}

		p.resampler = rs
	}

	return p, nil
}

// Process converts pcm at inSampleRate to 48 kHz super-resolved audio.
// inSampleRate must match the rate passed to NewFlashSRProcessor; the parameter
// is accepted for interface compatibility and validated against the configured rate.
func (p *FlashSRProcessor) Process(pcm []float32, inSampleRate int) ([]float32, int, error) {
	if len(pcm) == 0 {
		return nil, 0, errors.New("pockettts: input PCM is empty")
	}

	if inSampleRate != p.inRate {
		return nil, 0, fmt.Errorf("pockettts: got sample rate %d, processor configured for %d", inSampleRate, p.inRate)
	}

	if p.resampler != nil {
		var err error

		pcm, err = p.resampler.Process(pcm)
		if err != nil {
			return nil, 0, fmt.Errorf("pockettts: resample to 16kHz: %w", err)
		}
	}

	out, err := p.upsampler.Upsample16kTo48k(pcm)
	if err != nil {
		return nil, 0, fmt.Errorf("pockettts: upsample: %w", err)
	}

	return out, 48000, nil
}

// Int16ToFloat32 converts int16 PCM samples to float32 in [-1, 1].
// Values are divided by 32768.0 and clamped to [-1, 1].
func Int16ToFloat32(pcm []int16) []float32 {
	out := make([]float32, len(pcm))
	for i, v := range pcm {
		f := float32(v) / 32768.0
		if f > 1 {
			f = 1
		} else if f < -1 {
			f = -1
		}

		out[i] = f
	}

	return out
}
