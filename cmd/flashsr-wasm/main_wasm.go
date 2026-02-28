//go:build js && wasm

package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
	"syscall/js"

	"github.com/MeKo-Christian/flashsr-go/resample"
	"github.com/cwbudde/wav"
	goaudio "github.com/go-audio/audio"
)

const (
	targetSampleRate = 48000
	wavBitDepth      = 16
	wavNumChannels   = 1
	wavPCMFormat     = 1
)

type wavMeta struct {
	SampleRate  int
	NumChannels int
	BitDepth    int
}

var kernelFuncs []js.Func

func main() {
	api := js.Global().Get("Object").New()
	api.Set("version", "0.1.0-wasm")
	api.Set("targetSampleRate", targetSampleRate)
	kernelFuncs = append(kernelFuncs, js.FuncOf(analyzeWAVAsync))
	kernelFuncs = append(kernelFuncs, js.FuncOf(processWAVAsync))
	api.Set("analyzeWAV", kernelFuncs[0])
	api.Set("processWAV", kernelFuncs[1])

	js.Global().Set("FlashSRWebKernel", api)
	println("FlashSR web demo wasm kernel loaded")

	select {}
}

func analyzeWAVAsync(_ js.Value, args []js.Value) any {
	return newPromise(func(resolve, reject js.Value) {
		if len(args) < 1 {
			reject.Invoke("analyzeWAV requires a Uint8Array/ArrayBuffer WAV payload")
			return
		}

		wavBytes, err := copyJSBytes(args[0])
		if err != nil {
			reject.Invoke(err.Error())
			return
		}

		go func() {
			meta, err := inspectWAV(wavBytes)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}

			resolve.Invoke(js.ValueOf(map[string]any{
				"inputSampleRate":  meta.SampleRate,
				"targetSampleRate": targetSampleRate,
				"inputChannels":    meta.NumChannels,
				"inputBitDepth":    meta.BitDepth,
				"needsResample":    meta.SampleRate < targetSampleRate,
				"quality":          qualityLabel(meta.SampleRate),
				"message":          qualityMessage(meta.SampleRate),
			}))
		}()
	})
}

func processWAVAsync(_ js.Value, args []js.Value) any {
	return newPromise(func(resolve, reject js.Value) {
		if len(args) < 1 {
			reject.Invoke("processWAV requires a Uint8Array/ArrayBuffer WAV payload")
			return
		}

		wavBytes, err := copyJSBytes(args[0])
		if err != nil {
			reject.Invoke(err.Error())
			return
		}

		quality := parseQuality(args)

		go func() {
			meta, err := inspectWAV(wavBytes)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}

			if meta.SampleRate >= targetSampleRate {
				resolve.Invoke(js.ValueOf(map[string]any{
					"inputSampleRate":  meta.SampleRate,
					"outputSampleRate": meta.SampleRate,
					"inputChannels":    meta.NumChannels,
					"inputBitDepth":    meta.BitDepth,
					"needsResample":    false,
					"wasResampled":     false,
					"quality":          qualityLabel(meta.SampleRate),
					"message":          qualityMessage(meta.SampleRate),
					"resampleMode":     "not-required",
					"wavBytes":         jsUint8Array(wavBytes),
				}))
				return
			}

			samples, err := decodeWAVToMono(wavBytes)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}

			rs, err := resample.NewPolyphase(meta.SampleRate, targetSampleRate, resample.WithQuality(quality))
			if err != nil {
				reject.Invoke(fmt.Sprintf("build resampler %d->%d: %v", meta.SampleRate, targetSampleRate, err))
				return
			}

			out, err := rs.Process(samples)
			if err != nil {
				reject.Invoke(fmt.Sprintf("resample %d->%d: %v", meta.SampleRate, targetSampleRate, err))
				return
			}

			outWAV, err := encodeMonoWAV(out, targetSampleRate)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}

			resolve.Invoke(js.ValueOf(map[string]any{
				"inputSampleRate":  meta.SampleRate,
				"outputSampleRate": targetSampleRate,
				"inputChannels":    meta.NumChannels,
				"inputBitDepth":    meta.BitDepth,
				"needsResample":    true,
				"wasResampled":     true,
				"quality":          qualityLabel(meta.SampleRate),
				"message":          qualityMessage(meta.SampleRate),
				"resampleMode":     qualityName(quality),
				"inputSamples":     len(samples),
				"outputSamples":    len(out),
				"wavBytes":         jsUint8Array(outWAV),
			}))
		}()
	})
}

func newPromise(run func(resolve, reject js.Value)) js.Value {
	promise := js.Global().Get("Promise")

	var executor js.Func
	executor = js.FuncOf(func(_ js.Value, args []js.Value) any {
		defer executor.Release()
		if len(args) < 2 {
			return nil
		}

		run(args[0], args[1])
		return nil
	})

	return promise.New(executor)
}

func inspectWAV(wavBytes []byte) (wavMeta, error) {
	dec := wav.NewDecoder(bytes.NewReader(wavBytes))
	if !dec.IsValidFile() {
		return wavMeta{}, errors.New("invalid WAV file")
	}

	meta := wavMeta{
		SampleRate:  int(dec.SampleRate),
		NumChannels: int(dec.NumChans),
		BitDepth:    int(dec.BitDepth),
	}
	if meta.NumChannels <= 0 {
		meta.NumChannels = 1
	}
	if meta.SampleRate <= 0 {
		return wavMeta{}, fmt.Errorf("invalid sample rate %d", meta.SampleRate)
	}

	return meta, nil
}

func decodeWAVToMono(wavBytes []byte) ([]float32, error) {
	dec := wav.NewDecoder(bytes.NewReader(wavBytes))
	if !dec.IsValidFile() {
		return nil, errors.New("invalid WAV file")
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("decode WAV PCM: %w", err)
	}
	if buf == nil || len(buf.Data) == 0 {
		return nil, errors.New("decoded WAV has no PCM data")
	}

	channels := int(dec.NumChans)
	if buf.Format != nil && buf.Format.NumChannels > 0 {
		channels = buf.Format.NumChannels
	}
	if channels <= 0 {
		channels = 1
	}

	if channels == 1 {
		out := make([]float32, len(buf.Data))
		copy(out, buf.Data)
		return out, nil
	}

	return downmixMono(buf.Data, channels), nil
}

func encodeMonoWAV(samples []float32, sampleRate int) ([]byte, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("invalid output sample rate %d", sampleRate)
	}

	mem := &memoryWriteSeeker{}
	enc := wav.NewEncoder(mem, sampleRate, wavBitDepth, wavNumChannels, wavPCMFormat)

	pcm := &goaudio.Float32Buffer{
		Data: samples,
		Format: &goaudio.Format{
			SampleRate:  sampleRate,
			NumChannels: wavNumChannels,
		},
		SourceBitDepth: wavBitDepth,
	}

	if err := enc.Write(pcm); err != nil {
		return nil, fmt.Errorf("encode WAV PCM: %w", err)
	}
	if err := enc.Close(); err != nil {
		return nil, fmt.Errorf("close WAV encoder: %w", err)
	}

	return mem.Bytes(), nil
}

func downmixMono(interleaved []float32, channels int) []float32 {
	if channels <= 1 || len(interleaved) == 0 {
		out := make([]float32, len(interleaved))
		copy(out, interleaved)
		return out
	}

	frames := len(interleaved) / channels
	out := make([]float32, frames)

	for i := range frames {
		base := i * channels
		sum := float32(0)
		for ch := range channels {
			sum += interleaved[base+ch]
		}
		out[i] = sum / float32(channels)
	}

	return out
}

func copyJSBytes(v js.Value) ([]byte, error) {
	if v.IsUndefined() || v.IsNull() {
		return nil, errors.New("missing binary payload")
	}

	uint8Array := js.Global().Get("Uint8Array")
	arrayBuffer := js.Global().Get("ArrayBuffer")

	src := v
	if v.InstanceOf(arrayBuffer) {
		src = uint8Array.New(v)
	} else if !v.InstanceOf(uint8Array) {
		return nil, errors.New("payload must be Uint8Array or ArrayBuffer")
	}

	size := src.Get("byteLength").Int()
	if size == 0 {
		return nil, errors.New("payload is empty")
	}

	out := make([]byte, size)
	if n := js.CopyBytesToGo(out, src); n != size {
		return nil, fmt.Errorf("copied %d/%d bytes from JS payload", n, size)
	}

	return out, nil
}

func jsUint8Array(data []byte) js.Value {
	out := js.Global().Get("Uint8Array").New(len(data))
	_ = js.CopyBytesToJS(out, data)
	return out
}

func parseQuality(args []js.Value) resample.Quality {
	if len(args) < 2 || args[1].IsUndefined() || args[1].IsNull() {
		return resample.QualityBalanced
	}

	switch strings.ToLower(strings.TrimSpace(args[1].String())) {
	case "fast":
		return resample.QualityFast
	case "best":
		return resample.QualityBest
	default:
		return resample.QualityBalanced
	}
}

func qualityName(q resample.Quality) string {
	switch q {
	case resample.QualityFast:
		return "fast"
	case resample.QualityBest:
		return "best"
	default:
		return "balanced"
	}
}

func qualityLabel(sampleRate int) string {
	if sampleRate >= targetSampleRate {
		return "good"
	}
	if sampleRate >= 32000 {
		return "medium"
	}
	return "low"
}

func qualityMessage(sampleRate int) string {
	if sampleRate >= targetSampleRate {
		return "Input is already 48 kHz or higher; no resampling needed."
	}
	return "Input is below 48 kHz; FlashSR web demo will resample to 48 kHz."
}

type memoryWriteSeeker struct {
	buf []byte
	pos int64
}

func (m *memoryWriteSeeker) Write(p []byte) (int, error) {
	if m == nil {
		return 0, errors.New("nil memoryWriteSeeker")
	}
	if m.pos < 0 {
		return 0, errors.New("negative write position")
	}

	end := m.pos + int64(len(p))
	if end > int64(len(m.buf)) {
		newBuf := make([]byte, end)
		copy(newBuf, m.buf)
		m.buf = newBuf
	}

	copy(m.buf[m.pos:end], p)
	m.pos = end
	return len(p), nil
}

func (m *memoryWriteSeeker) Seek(offset int64, whence int) (int64, error) {
	if m == nil {
		return 0, errors.New("nil memoryWriteSeeker")
	}

	var next int64
	switch whence {
	case io.SeekStart:
		next = offset
	case io.SeekCurrent:
		next = m.pos + offset
	case io.SeekEnd:
		next = int64(len(m.buf)) + offset
	default:
		return 0, fmt.Errorf("invalid whence %d", whence)
	}

	if next < 0 {
		return 0, errors.New("negative seek position")
	}

	m.pos = next
	return m.pos, nil
}

func (m *memoryWriteSeeker) Bytes() []byte {
	out := make([]byte, len(m.buf))
	copy(out, m.buf)
	return out
}
