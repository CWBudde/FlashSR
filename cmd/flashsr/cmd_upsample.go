package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/stream"
	"github.com/spf13/cobra"
)

const inputSampleRate = 16000
const outputSampleRate = 48000

type upsampleFlags struct {
	input     string
	output    string
	modelPath string
	ortLib    string
	threads   int
	streaming bool
	chunkSize int
	inputRate int // 0 = auto-detect from WAV header
}

func newUpsampleCmd() *cobra.Command {
	var f upsampleFlags

	cmd := &cobra.Command{
		Use:   "upsample",
		Short: "Upsample a 16 kHz WAV file to 48 kHz",
		Long: `upsample reads a 16 kHz PCM WAV file, runs FlashSR super-resolution,
and writes a 48 kHz WAV output.

Example:
  flashsr upsample --input speech.wav --output speech_48k.wav \
    --ort-lib /usr/lib/libonnxruntime.so.1.24.1`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runUpsample(f)
		},
	}

	cmd.Flags().StringVarP(&f.input, "input", "i", "", "input WAV file (16 kHz, required)")
	cmd.Flags().StringVarP(&f.output, "output", "o", "", "output WAV file (48 kHz, required)")
	cmd.Flags().StringVar(&f.modelPath, "model-path", "", "override embedded ONNX model")
	cmd.Flags().StringVar(&f.ortLib, "ort-lib", "", "path to libonnxruntime shared library")
	cmd.Flags().IntVar(&f.threads, "threads", 1, "ORT intra/inter thread count")
	cmd.Flags().BoolVar(&f.streaming, "stream", false, "use streaming (chunk-based) mode")
	cmd.Flags().IntVar(&f.chunkSize, "chunk-size", 4000, "chunk size in samples (streaming mode)")
	cmd.Flags().IntVar(&f.inputRate, "input-rate", 0, "input sample rate override (0 = read from WAV header)")

	_ = cmd.MarkFlagRequired("input")
	_ = cmd.MarkFlagRequired("output")

	return cmd
}

// runUpsample is the top-level handler: validates flags, builds an Upsampler,
// then delegates to runUpsampleWithUpsampler.
func runUpsample(f upsampleFlags) error {
	if _, err := os.Stat(f.input); errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("input file not found: %s", f.input)
	}

	// Read WAV header to determine sample rate.
	_, sr, err := readWAV(f.input)
	if err != nil {
		return fmt.Errorf("read input: %w", err)
	}

	// --input-rate overrides the WAV header.
	if f.inputRate != 0 {
		sr = f.inputRate
	}

	// Reject rates that are neither 16kHz nor supported for resampling.
	if sr != inputSampleRate && sr <= 0 {
		return fmt.Errorf("invalid input sample rate %d Hz", sr)
	}

	u, err := flashsr.New(flashsr.Config{
		ModelPath:       f.modelPath,
		ORTLibPath:      f.ortLib,
		NumThreadsIntra: f.threads,
		NumThreadsInter: f.threads,
		InputRate:       sr,
	})
	if err != nil {
		return fmt.Errorf("init engine: %w", err)
	}
	defer u.Close()

	return runUpsampleWithUpsampler(u, f)
}

// runUpsampleWithUpsampler performs the actual WAV decode → upsample → WAV
// encode pipeline. It accepts a pre-built Upsampler so tests can inject a mock.
func runUpsampleWithUpsampler(u *flashsr.Upsampler, f upsampleFlags) error {
	pcm, _, err := readWAV(f.input)
	if err != nil {
		return fmt.Errorf("read input WAV: %w", err)
	}

	var out []float32

	if f.streaming {
		out, err = upsampleStreaming(u, pcm, f.chunkSize)
	} else {
		out, err = u.Upsample16kTo48k(pcm)
	}

	if err != nil {
		return fmt.Errorf("upsample: %w", err)
	}

	if err := writeWAV(f.output, out, outputSampleRate); err != nil {
		return fmt.Errorf("write output WAV: %w", err)
	}

	fmt.Fprintf(os.Stdout, "%s → %s  (%d → %d samples)\n",
		f.input, f.output, len(pcm), len(out))

	return nil
}

// upsampleStreaming processes pcm in chunks via stream.Streamer and collects
// all output samples.
func upsampleStreaming(u *flashsr.Upsampler, pcm []float32, chunkSize int) ([]float32, error) {
	st := stream.New(u.Engine(), stream.Config{ChunkSize: chunkSize})
	defer st.Reset()

	// Feed input in chunkSize pieces.
	for i := 0; i < len(pcm); i += chunkSize {
		end := i + chunkSize
		if end > len(pcm) {
			end = len(pcm)
		}

		if err := st.Write(pcm[i:end]); err != nil {
			return nil, fmt.Errorf("stream write: %w", err)
		}
	}

	// Flush any remaining partial chunk.
	if err := st.Flush(); err != nil {
		return nil, fmt.Errorf("stream flush: %w", err)
	}

	// Drain output buffer.
	out := make([]float32, st.Buffered())
	n, err := st.Read(out)

	if err != nil {
		return nil, fmt.Errorf("stream read: %w", err)
	}

	return out[:n], nil
}
