//go:build !algodsp

package resample

import "math"

// linearResampler is a stateful linear interpolating sample-rate converter.
// It preserves fractional phase across Process calls so chunk boundaries are seamless.
type linearResampler struct {
	ratio float64 // inRate / outRate
	phase float64 // current fractional position in input stream
	prev  float32 // last sample of previous chunk (for inter-chunk interpolation)
}

func newResampler(inRate, outRate int) (Resampler, error) {
	return &linearResampler{
		ratio: float64(inRate) / float64(outRate),
	}, nil
}

func (r *linearResampler) Process(in []float32) ([]float32, error) {
	if len(in) == 0 {
		return nil, nil
	}

	// Estimate output length and pre-allocate.
	outLen := int(math.Ceil(float64(len(in)) / r.ratio))
	out := make([]float32, 0, outLen)

	for {
		// Integer and fractional parts of the current read position.
		idx := int(r.phase)
		frac := r.phase - float64(idx)

		if idx >= len(in) {
			break
		}

		var a, b float32
		if idx == 0 {
			a = r.prev
		} else {
			a = in[idx-1]
		}

		b = in[idx]

		out = append(out, a+float32(frac)*(b-a))
		r.phase += r.ratio
	}

	// Carry over: shift phase relative to the consumed input.
	r.phase -= float64(len(in))
	if len(in) > 0 {
		r.prev = in[len(in)-1]
	}

	return out, nil
}

func (r *linearResampler) Reset() {
	r.phase = 0
	r.prev = 0
}
