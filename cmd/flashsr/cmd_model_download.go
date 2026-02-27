package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/MeKo-Christian/flashsr-go/model"
	"github.com/spf13/cobra"
)

func newModelDownloadCmd() *cobra.Command {
	var hfRepo string
	var revision string
	var filename string
	var outPath string
	var hfToken string

	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download the FlashSR ONNX model from Hugging Face",
		Long: `Download the FlashSR ONNX model from Hugging Face and save it locally.

The SHA256 printed after a successful download can be used to pin the model
in model.go (ExpectedSHA256 constant) for reproducible builds.

Environment variables:
  HF_TOKEN   Hugging Face API token (alternative to --hf-token)`,
		RunE: func(_ *cobra.Command, _ []string) error {
			if hfToken == "" {
				hfToken = os.Getenv("HF_TOKEN")
			}

			result, err := model.Download(model.DownloadOptions{
				Repo:     hfRepo,
				Revision: revision,
				Filename: filename,
				OutPath:  outPath,
				HFToken:  hfToken,
				Stdout:   os.Stdout,
			})
			if err != nil {
				var denied *model.AccessDeniedError
				if errors.As(err, &denied) {
					_, _ = fmt.Fprintln(os.Stderr, "hint: set HF_TOKEN or pass --hf-token if the repo is gated")
				}

				return fmt.Errorf("model download: %w", err)
			}

			if result.Skipped {
				fmt.Printf("model already up-to-date: %s\n", result.Path)
			} else {
				fmt.Printf("model saved:  %s\n", result.Path)
				fmt.Printf("sha256:       %s\n", result.SHA256)
				fmt.Println()
				fmt.Println("To pin this model, set ExpectedSHA256 in model/model.go:")
				fmt.Printf("  const ExpectedSHA256 = %q\n", result.SHA256)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&hfRepo, "hf-repo", model.DefaultRepo, "Hugging Face repository")
	cmd.Flags().StringVar(&revision, "revision", model.DefaultRevision, "Branch, tag, or commit")
	cmd.Flags().StringVar(&filename, "filename", model.DefaultFilename, "File path within the repository")
	cmd.Flags().StringVarP(&outPath, "out", "o", "assets/model.onnx", "Local output path")
	cmd.Flags().StringVar(&hfToken, "hf-token", "", "Hugging Face token (falls back to HF_TOKEN env)")

	return cmd
}
