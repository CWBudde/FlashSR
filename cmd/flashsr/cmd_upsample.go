package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/stream"
)

type upsampleFlags struct {
	input     string
	output    string
	modelPath string
	ortLib    string
	threads   int
	streaming bool
	chunkSize int
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

	_ = cmd.MarkFlagRequired("input")
	_ = cmd.MarkFlagRequired("output")

	return cmd
}

func runUpsample(f upsampleFlags) error {
	// Validate input file exists.
	if _, err := os.Stat(f.input); errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("input file not found: %s", f.input)
	}

	// Build upsampler.
	u, err := flashsr.New(flashsr.Config{
		ModelPath:       f.modelPath,
		ORTLibPath:      f.ortLib,
		NumThreadsIntra: f.threads,
		NumThreadsInter: f.threads,
	})
	if err != nil {
		return fmt.Errorf("init: %w", err)
	}
	defer u.Close()

	// TODO(phase4): decode input WAV → []float32 via cwbudde/wav.
	// For now, print a placeholder.
	fmt.Fprintf(os.Stderr, "reading %s ...\n", f.input)
	fmt.Fprintf(os.Stderr, "NOTE: WAV I/O not yet wired (Phase 4)\n")

	if f.streaming {
		st := stream.New(nil, stream.Config{ // nil engine placeholder
			ChunkSize: f.chunkSize,
		})
		_ = st
		fmt.Fprintln(os.Stderr, "streaming mode selected")
	}

	// TODO(phase4): write output WAV.
	fmt.Fprintf(os.Stderr, "would write 48 kHz output to %s\n", f.output)
	return nil
}
