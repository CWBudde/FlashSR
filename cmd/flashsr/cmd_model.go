package main

import "github.com/spf13/cobra"

func newModelCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Model acquisition commands",
	}

	cmd.AddCommand(newModelDownloadCmd())

	return cmd
}
