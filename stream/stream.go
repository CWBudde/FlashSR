// Package stream provides a streaming wrapper around a FlashSR Engine.
// It implements the upstream Python behavior: 500-sample input overlap,
// linear crossfade, and first-chunk output trimming.
package stream

import (
	"errors"
	"fmt"
	"io"

	"github.com/cwbudde/flashsr-go/engine"
)

const (
	defaultChunkSize  = 4000   // samples @ 16 kHz = 250 ms
	defaultOverlap    = 500    // input samples prepended from previous chunk
	defaultBufferCap  = 480000 // 30 s @ 16 kHz
	firstChunkTrim    = 2000   // output samples trimmed from first yield
	outputOverlapSkip = 1000   // upsampled[1000:] — alignment offset from upstream
)

// Config controls the Streamer parameters.
type Config struct {
	// ChunkSize is the number of 16 kHz input samples per inference call (default: 4000).
	ChunkSize int

	// Overlap is the number of input samples prepended from the previous chunk (default: 500).
	Overlap int

	// BufferCap is the maximum input buffer capacity in samples (default: 480000 = 30 s).
	BufferCap int
}

func (c Config) chunkSize() int {
	if c.ChunkSize > 0 {
		return c.ChunkSize
	}

	return defaultChunkSize
}

func (c Config) overlap() int {
	if c.Overlap > 0 {
		return c.Overlap
	}

	return defaultOverlap
}

//nolint:unused // Reserved for future backpressure / buffer limiting.
func (c Config) bufferCap() int {
	if c.BufferCap > 0 {
		return c.BufferCap
	}

	return defaultBufferCap
}

// Streamer buffers input samples, runs inference chunk-by-chunk, and emits
// 48 kHz output with overlap-based crossfade stitching.
type Streamer struct {
	eng       engine.Engine
	cfg       Config
	inputBuf  []float32 // pending input samples not yet processed
	overlapIn []float32 // last cfg.Overlap() samples of the previous input chunk
	outputBuf []float32 // processed output waiting to be Read

	// crossfade state
	prevTail   []float32 // last outOverlap samples of previous output chunk
	firstChunk bool
}

// New creates a Streamer wrapping the given Engine.
func New(eng engine.Engine, cfg Config) *Streamer {
	return &Streamer{
		eng:        eng,
		cfg:        cfg,
		firstChunk: true,
	}
}

// Write pushes new 16 kHz input samples into the streamer and processes any
// complete chunks. Processed output is buffered and available via Read.
func (s *Streamer) Write(samples []float32) error {
	s.inputBuf = append(s.inputBuf, samples...)
	return s.drainChunks()
}

// Read pops up to len(out) output samples from the internal buffer.
// Returns (n, io.EOF) when Flush has been called and the buffer is empty.
func (s *Streamer) Read(out []float32) (int, error) {
	n := copy(out, s.outputBuf)

	s.outputBuf = s.outputBuf[n:]
	if n == 0 {
		return 0, io.EOF
	}

	return n, nil
}

// Flush processes any remaining buffered input (zero-padded to a full chunk).
// Call after the last Write; then drain output with Read.
func (s *Streamer) Flush() error {
	chunk := s.cfg.chunkSize()
	if len(s.inputBuf) == 0 {
		return nil
	}
	// Zero-pad to full chunk size.
	for len(s.inputBuf) < chunk {
		s.inputBuf = append(s.inputBuf, 0)
	}

	return s.drainChunks()
}

// Reset clears all internal state so the Streamer can be reused for a new stream.
func (s *Streamer) Reset() {
	s.inputBuf = s.inputBuf[:0]
	s.overlapIn = nil
	s.outputBuf = s.outputBuf[:0]
	s.prevTail = nil
	s.firstChunk = true
}

// Buffered returns the number of output samples ready to be Read.
func (s *Streamer) Buffered() int { return len(s.outputBuf) }

// drainChunks processes all complete chunks from inputBuf.
func (s *Streamer) drainChunks() error {
	chunk := s.cfg.chunkSize()
	for len(s.inputBuf) >= chunk {
		err := s.processChunk(s.inputBuf[:chunk])
		if err != nil {
			return err
		}

		s.inputBuf = s.inputBuf[chunk:]
	}

	return nil
}

// processChunk runs one inference step and appends the result to outputBuf.
// Implements the upstream Python streaming algorithm:
//  1. Prepend overlapIn to the chunk (provides model context at boundaries).
//  2. Run inference on the combined input.
//  3. Skip the first outputOverlapSkip (1000) output samples (alignment).
//  4. Apply linear crossfade over outOverlap (overlap*3) samples with the previous tail.
//  5. On the first chunk, trim firstChunkTrim (2000) output samples.
func (s *Streamer) processChunk(chunk []float32) error {
	overlap := s.cfg.overlap()

	// 1. Build model input: [overlapIn | chunk].
	modelInput := make([]float32, 0, len(s.overlapIn)+len(chunk))
	modelInput = append(modelInput, s.overlapIn...)
	modelInput = append(modelInput, chunk...)

	// 2. Update overlap buffer for next call.
	if len(chunk) >= overlap {
		s.overlapIn = clone(chunk[len(chunk)-overlap:])
	} else {
		s.overlapIn = clone(chunk)
	}

	// 3. Inference.
	raw, err := s.eng.Run(modelInput)
	if err != nil {
		return fmt.Errorf("stream: engine run: %w", err)
	}

	// 4. Alignment: skip first outputOverlapSkip output samples.
	if len(raw) > outputOverlapSkip {
		raw = raw[outputOverlapSkip:]
	}

	// 5. Crossfade with previous tail.
	outOverlap := overlap * 3 // 3× upsample ratio
	aligned := s.applyCrossfade(raw, outOverlap)

	// 6. Update tail for next crossfade.
	if len(aligned) >= outOverlap {
		s.prevTail = clone(aligned[len(aligned)-outOverlap:])
	} else {
		s.prevTail = clone(aligned)
	}

	// 7. First-chunk trimming.
	if s.firstChunk {
		if len(aligned) > firstChunkTrim {
			aligned = aligned[firstChunkTrim:]
		} else {
			aligned = nil
		}

		s.firstChunk = false
	}

	s.outputBuf = append(s.outputBuf, aligned...)

	return nil
}

// applyCrossfade blends the tail of the previous output with the head of new.
// Returns the merged output (previous tail replaced by the crossfade region).
func (s *Streamer) applyCrossfade(next []float32, overlapOut int) []float32 {
	if len(s.prevTail) == 0 {
		return next
	}

	if len(next) == 0 {
		return next
	}

	n := min(min(overlapOut, len(s.prevTail)), len(next))

	out := make([]float32, len(next))
	copy(out, next)

	for i := range n {
		t := float32(i) / float32(n)
		out[i] = s.prevTail[i]*(1-t) + next[i]*t
	}

	return out
}

func clone(s []float32) []float32 {
	if s == nil {
		return nil
	}

	c := make([]float32, len(s))
	copy(c, s)

	return c
}

// Ensure errors package is used (io.EOF reference above).
var _ = errors.New
