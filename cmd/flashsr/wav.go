package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/cwbudde/wav"
	goaudio "github.com/go-audio/audio"
)

const (
	wavBitDepth  = 16
	wavNumChans  = 1 // mono
	wavPCMFormat = 1 // PCM
)

// readWAV opens a WAV file and returns its float32 PCM samples and sample rate.
// The caller should verify the sample rate matches their expectations.
func readWAV(path string) (samples []float32, sampleRate int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open WAV: %w", err)
	}

	defer func() { _ = f.Close() }()

	dec := wav.NewDecoder(f)
	if !dec.IsValidFile() {
		return nil, 0, errors.New("not a valid WAV file")
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("decode WAV PCM: %w", err)
	}

	return buf.Data, int(dec.SampleRate), nil
}

// writeWAV encodes float32 PCM samples as a 16-bit mono WAV file.
func writeWAV(path string, samples []float32, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create WAV: %w", err)
	}

	enc := wav.NewEncoder(f, sampleRate, wavBitDepth, wavNumChans, wavPCMFormat)

	buf := &goaudio.Float32Buffer{
		Data: samples,
		Format: &goaudio.Format{
			SampleRate:  sampleRate,
			NumChannels: wavNumChans,
		},
		SourceBitDepth: wavBitDepth,
	}

	err = enc.Write(buf)
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("encode WAV PCM: %w", err)
	}

	err = enc.Close()
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("close WAV encoder: %w", err)
	}

	return f.Close()
}
