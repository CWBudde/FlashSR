// Package assets embeds the FlashSR ONNX model binary.
package assets

import _ "embed"

// ModelONNX contains the FlashSR ONNX model compiled into the binary.
//
//go:embed model.onnx
var ModelONNX []byte
