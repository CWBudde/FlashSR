// Package engine defines the inference backend interface for FlashSR.
package engine

// EngineInfo describes the loaded model and runtime configuration.
//
//nolint:revive // Public API: engine.EngineInfo is intentionally explicit.
type EngineInfo struct {
	InputName  string
	OutputName string
	InputRank  int    // number of tensor dimensions (2 = [1,N], 3 = [1,1,N])
	Provider   string // "CPU", "CUDA", "CoreML", etc.
	OrtVersion string
}

// Engine is the minimal interface for running audio super-resolution inference.
// Implementations must be safe to call Run concurrently.
type Engine interface {
	// Run performs inference. Input is float32 PCM in [-1, 1].
	// Returns the upsampled output or an error.
	Run(input []float32) ([]float32, error)
	Close() error
	Info() EngineInfo
}
