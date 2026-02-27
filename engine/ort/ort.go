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
	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error

	// ErrNoORTLib is returned when no shared library path is configured.
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

	// InputName overrides the auto-detected tensor input name.
	InputName string

	// OutputName overrides the auto-detected tensor output name.
	OutputName string

	// UpsampleRatio is the expected output/input length ratio (default: 3 for 16→48 kHz).
	UpsampleRatio int
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

func (c *Config) upsampleRatio() int {
	if c.UpsampleRatio > 0 {
		return c.UpsampleRatio
	}

	return 3
}

// Engine is an ONNX Runtime backed inference engine.
type Engine struct {
	session       *ort.DynamicAdvancedSession
	inputName     string
	outputName    string
	inputRank     int // 2 = [1,N], 3 = [1,1,N]
	outputRank    int
	upsampleRatio int
}

// New creates an ORT Engine from the given model bytes.
// The ORT shared library must be accessible via Config.LibraryPath or FLASHSR_ORT_LIB.
// ORT session initialization is expensive; call New once per process and reuse.
func New(modelBytes []byte, cfg Config) (*Engine, error) {
	libPath := cfg.libraryPath()
	if libPath == "" {
		return nil, ErrNoORTLib
	}

	if len(modelBytes) == 0 {
		return nil, errors.New("ort: model bytes are empty")
	}

	// ORT environment must be initialized exactly once per process.
	// SetSharedLibraryPath must be called before InitializeEnvironment.
	initOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)

		initErr = ort.InitializeEnvironment(ort.WithLogLevelError())
	})

	if initErr != nil {
		return nil, fmt.Errorf("ort: environment init: %w", initErr)
	}

	// Introspect model to discover tensor names and shapes.
	inputInfos, outputInfos, err := ort.GetInputOutputInfoWithONNXData(modelBytes)
	if err != nil {
		return nil, fmt.Errorf("ort: introspect model: %w", err)
	}

	if len(inputInfos) == 0 || len(outputInfos) == 0 {
		return nil, errors.New("ort: model has no inputs or outputs")
	}

	inputName := cfg.InputName
	if inputName == "" {
		inputName = inputInfos[0].Name
	}

	outputName := cfg.OutputName
	if outputName == "" {
		outputName = outputInfos[0].Name
	}

	inputRank := len(inputInfos[0].Dimensions)
	if inputRank == 0 {
		inputRank = 3 // safe default: [1, 1, N]
	}

	outputRank := len(outputInfos[0].Dimensions)
	if outputRank == 0 {
		outputRank = inputRank
	}

	// Build session options.
	intra, inter := cfg.threads()

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("ort: session options: %w", err)
	}
	defer opts.Destroy()

	if err := opts.SetIntraOpNumThreads(intra); err != nil {
		return nil, fmt.Errorf("ort: set intra threads: %w", err)
	}

	if err := opts.SetInterOpNumThreads(inter); err != nil {
		return nil, fmt.Errorf("ort: set inter threads: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelBytes,
		[]string{inputName},
		[]string{outputName},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("ort: create session: %w", err)
	}

	return &Engine{
		session:       session,
		inputName:     inputName,
		outputName:    outputName,
		inputRank:     inputRank,
		outputRank:    outputRank,
		upsampleRatio: cfg.upsampleRatio(),
	}, nil
}

// Run executes inference on input PCM samples.
// Input is float32 PCM in [-1, 1]. Returns upsampled float32 PCM.
// Safe to call concurrently on the same Engine.
func (e *Engine) Run(input []float32) ([]float32, error) {
	n := int64(len(input))

	inputTensor, err := ort.NewTensor(e.inputShape(n), input)
	if err != nil {
		return nil, fmt.Errorf("ort: create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outN := n * int64(e.upsampleRatio)

	outputTensor, err := ort.NewEmptyTensor[float32](e.outputShape(outN))
	if err != nil {
		return nil, fmt.Errorf("ort: create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	if err := e.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	); err != nil {
		return nil, fmt.Errorf("ort: run: %w", err)
	}

	// Copy output data; the tensor's slice is invalidated after Destroy.
	src := outputTensor.GetData()
	result := make([]float32, len(src))
	copy(result, src)

	return result, nil
}

// Close releases the ORT session.
func (e *Engine) Close() error {
	return e.session.Destroy()
}

// Info returns metadata about the loaded model and runtime.
func (e *Engine) Info() engine.EngineInfo {
	return engine.EngineInfo{
		InputName:  e.inputName,
		OutputName: e.outputName,
		InputRank:  e.inputRank,
		Provider:   "CPU",
		OrtVersion: ort.GetVersion(),
	}
}

// inputShape builds the tensor shape for a given number of samples.
func (e *Engine) inputShape(n int64) ort.Shape {
	switch e.inputRank {
	case 2:
		return ort.NewShape(1, n)
	default: // 3
		return ort.NewShape(1, 1, n)
	}
}

// outputShape builds the expected output tensor shape.
func (e *Engine) outputShape(n int64) ort.Shape {
	switch e.outputRank {
	case 2:
		return ort.NewShape(1, n)
	default: // 3
		return ort.NewShape(1, 1, n)
	}
}

// Compile-time interface check.
var _ engine.Engine = (*Engine)(nil)
