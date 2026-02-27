package flashsr

import "errors"

var (
	ErrModelLoad   = errors.New("flashsr: model load failed")
	ErrEngineInit  = errors.New("flashsr: engine init failed")
	ErrInferFailed = errors.New("flashsr: inference failed")
)
