// Command flashsr is the CLI for FlashSR audio super-resolution.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

func main() {
	err := rootCmd().Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func rootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "flashsr",
		Short: "FlashSR audio super-resolution: 16 kHz → 48 kHz",
		Long: `flashsr converts WAV audio from 16 kHz to 48 kHz using a small ONNX model.

Environment variables:
  FLASHSR_ORT_LIB      Path to libonnxruntime shared library
  FLASHSR_MODEL_PATH   Override the embedded ONNX model file`,
		SilenceUsage: true,
	}

	cmd.AddCommand(newUpsampleCmd())
	cmd.AddCommand(newDoctorCmd())

	return cmd
}
