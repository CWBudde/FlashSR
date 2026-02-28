package main

import (
	"fmt"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/model"
	"github.com/spf13/cobra"
)

type doctorFlags struct {
	ortLib    string
	modelPath string
}

func newDoctorCmd() *cobra.Command {
	var f doctorFlags

	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Verify the ORT library and embedded model",
		Long: `doctor checks that the ORT shared library and FlashSR model are correctly
configured and prints version/diagnostic information.

Example:
  flashsr doctor --ort-lib /usr/lib/libonnxruntime.so.1.24.1`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runDoctor(cmd, f)
		},
	}

	cmd.Flags().StringVar(&f.ortLib, "ort-lib", "", "path to libonnxruntime shared library")
	cmd.Flags().StringVar(&f.modelPath, "model-path", "", "override embedded ONNX model")

	return cmd
}

func runDoctor(cmd *cobra.Command, f doctorFlags) error {
	out := cmd.OutOrStdout()

	_, _ = fmt.Fprintln(out, "=== FlashSR Doctor ===")

	// --- Model check ---
	_, _ = fmt.Fprintln(out, "\n[Model]")

	modelBytes, err := model.Load(model.Config{Path: f.modelPath})
	if err != nil {
		_, _ = fmt.Fprintf(out, "  ✗ model load failed: %v\n", err)
	} else {
		_, _ = fmt.Fprintf(out, "  ✓ model loaded (%d bytes)\n", len(modelBytes))

		if model.ExpectedSHA256 != "" {
			_, _ = fmt.Fprintf(out, "  ✓ pinned SHA256: %s\n", model.ExpectedSHA256)
		} else {
			_, _ = fmt.Fprintln(out, "  ⚠ SHA256 not yet pinned (Phase 2 TODO)")
		}
	}

	// --- ORT / Engine check ---
	_, _ = fmt.Fprintln(out, "\n[ORT Engine]")

	u, err := flashsr.New(flashsr.Config{
		ModelPath:  f.modelPath,
		ORTLibPath: f.ortLib,
	})
	if err != nil {
		_, _ = fmt.Fprintf(out, "  ✗ engine init failed: %v\n", err)
		_, _ = fmt.Fprintln(out, "\nDoctor finished with errors.")

		return nil // non-fatal; show all results
	}

	defer func() { _ = u.Close() }()

	info := u.EngineInfo()
	_, _ = fmt.Fprintf(out, "  ✓ ORT version:  %s\n", info.OrtVersion)
	_, _ = fmt.Fprintf(out, "  ✓ provider:     %s\n", info.Provider)
	_, _ = fmt.Fprintf(out, "  ✓ input tensor: %s (rank %d)\n", info.InputName, info.InputRank)
	_, _ = fmt.Fprintf(out, "  ✓ output tensor: %s\n", info.OutputName)

	_, _ = fmt.Fprintln(out, "\nDoctor finished — all checks passed.")

	return nil
}
