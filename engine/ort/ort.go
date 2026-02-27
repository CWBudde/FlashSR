// Package ort provides an ONNX Runtime backed Engine implementation.
//
// It uses github.com/yalue/onnxruntime_go which loads libonnxruntime dynamically
// via dlopen/LoadLibrary. cgo is required.
//
// Set the shared library path via Config.LibraryPath or the FLASHSR_ORT_LIB
// environment variable before calling New.
package ort

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/MeKo-Christian/flashsr-go/engine"
)

var (
	initOnce    sync.Once
	initErr     error
	ErrNoORTLib = errors.New("ort: shared library path is required (set Config.LibraryPath or FLASHSR_ORT_LIB)")
)

// Config configures the ORT engine.
type Config struct {
	// LibraryPath is the path to the ORT shared library (.so/.dylib/.dll).
	// Falls back to FLASHSR_ORT_LIB env var when empty.
	LibraryPath string

	// NumThreadsIntra controls intra-op parallelism (default: 1).
	NumThreadsIntra int

	// NumThreadsInter controls inter-op parallelism (default: 1).
	NumThreadsInter int

	// InputName overrides auto-detected tensor input name (e.g. "x" or "audio_values").
	InputName string

	// OutputName overrides auto-detected tensor output name.
	OutputName string
}

func (c *Config) libraryPath() string {
	if c.LibraryPath != "" {
		return c.LibraryPath
	}
	return os.Getenv("FLASHSR_ORT_LIB")
}

func (c *Config) threads() (intra, inter int) {
	intra = c.NumThreadsIntra
	if intra <= 0 {
		intra = 1
	}
	inter = c.NumThreadsInter
	if inter <= 0 {
		inter = 1
	}
	return
}

// Engine is an ONNX Runtime backed inference engine.
type Engine struct {
	cfg        Config
	inputName  string
	outputName string
	inputRank  int
	// session and related ORT objects will be added when yalue/onnxruntime_go is wired in.
	// Placeholder so the package compiles without the dependency for now.
}

// New creates an ORT engine from the given model bytes.
// The ORT shared library must be accessible via Config.LibraryPath or FLASHSR_ORT_LIB.
func New(modelBytes []byte, cfg Config) (*Engine, error) {
	libPath := cfg.libraryPath()
	if libPath == "" {
		return nil, ErrNoORTLib
	}

	if len(modelBytes) == 0 {
		return nil, errors.New("ort: model bytes are empty")
	}

	// ORT environment must be initialized exactly once per process.
	initOnce.Do(func() {
		initErr = initORT(libPath)
	})
	if initErr != nil {
		return nil, fmt.Errorf("ort: environment init: %w", initErr)
	}

	inputName := cfg.InputName
	outputName := cfg.OutputName

	// TODO(phase1): create ORT session from modelBytes, detect tensor names/rank.
	// For now, apply sensible defaults so the struct is usable in tests.
	if inputName == "" {
		inputName = "x"
	}
	if outputName == "" {
		outputName = "output"
	}

	return &Engine{
		cfg:        cfg,
		inputName:  inputName,
		outputName: outputName,
		inputRank:  3, // [1, 1, N] — most common in streaming code
	}, nil
}

// Run executes inference on input PCM samples.
func (e *Engine) Run(input []float32) ([]float32, error) {
	// TODO(phase1): reshape input according to e.inputRank, call ORT session.Run.
	return nil, errors.New("ort: Run not yet implemented — wire yalue/onnxruntime_go in Phase 1")
}

// Close releases ORT session resources.
func (e *Engine) Close() error {
	// TODO(phase1): session.Destroy()
	return nil
}

// Info returns metadata about the loaded model and runtime.
func (e *Engine) Info() engine.EngineInfo {
	return engine.EngineInfo{
		InputName:  e.inputName,
		OutputName: e.outputName,
		InputRank:  e.inputRank,
		Provider:   "CPU",
		OrtVersion: ortVersion(),
	}
}

// initORT loads the shared library and initializes the ORT environment.
func initORT(libPath string) error {
	// TODO(phase1): ort.SetSharedLibraryPath(libPath); ort.InitializeEnvironment()
	_ = libPath
	return nil
}

// ortVersion returns the ORT runtime version string.
func ortVersion() string {
	// TODO(phase1): return ort.GetVersionString()
	return "not yet initialized"
}

// Compile-time interface check.
var _ engine.Engine = (*Engine)(nil)
