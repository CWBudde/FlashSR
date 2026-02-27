package main

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/MeKo-Christian/flashsr-go/flashsr"
	"github.com/MeKo-Christian/flashsr-go/model"
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
		RunE: func(cmd *cobra.Command, args []string) error {
			return runDoctor(f)
		},
	}

	cmd.Flags().StringVar(&f.ortLib, "ort-lib", "", "path to libonnxruntime shared library")
	cmd.Flags().StringVar(&f.modelPath, "model-path", "", "override embedded ONNX model")

	return cmd
}

func runDoctor(f doctorFlags) error {
	fmt.Println("=== FlashSR Doctor ===")

	// --- Model check ---
	fmt.Println("\n[Model]")
	modelBytes, err := model.Load(model.Config{Path: f.modelPath})
	if err != nil {
		fmt.Printf("  ✗ model load failed: %v\n", err)
	} else {
		fmt.Printf("  ✓ model loaded (%d bytes)\n", len(modelBytes))
		if model.ExpectedSHA256 != "" {
			fmt.Printf("  ✓ pinned SHA256: %s\n", model.ExpectedSHA256)
		} else {
			fmt.Println("  ⚠ SHA256 not yet pinned (Phase 2 TODO)")
		}
	}

	// --- ORT / Engine check ---
	fmt.Println("\n[ORT Engine]")
	u, err := flashsr.New(flashsr.Config{
		ModelPath:  f.modelPath,
		ORTLibPath: f.ortLib,
	})
	if err != nil {
		fmt.Printf("  ✗ engine init failed: %v\n", err)
		fmt.Println("\nDoctor finished with errors.")
		return nil // non-fatal; show all results
	}
	defer u.Close()

	info := u.EngineInfo()
	fmt.Printf("  ✓ ORT version:  %s\n", info.OrtVersion)
	fmt.Printf("  ✓ provider:     %s\n", info.Provider)
	fmt.Printf("  ✓ input tensor: %s (rank %d)\n", info.InputName, info.InputRank)
	fmt.Printf("  ✓ output tensor: %s\n", info.OutputName)

	fmt.Println("\nDoctor finished — all checks passed.")
	return nil
}
