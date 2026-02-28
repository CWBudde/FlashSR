package testutil

import (
	"errors"
	"math"
)

// RMSError computes the RMS error between a and b in decibels.
// Returns math.Inf(-1) when both signals are silent.
// Returns an error when slices differ in length.
func RMSError(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("testutil: RMSError: slices must be the same length")
	}

	if len(a) == 0 {
		return math.Inf(-1), nil
	}

	var sum float64

	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}

	rms := math.Sqrt(sum / float64(len(a)))
	if rms == 0 {
		return math.Inf(-1), nil
	}

	return 20 * math.Log10(rms), nil
}

// RMS returns the root-mean-square level of a in decibels (relative to full scale).
func RMS(a []float32) float64 {
	if len(a) == 0 {
		return math.Inf(-1)
	}

	var sum float64
	for _, v := range a {
		sum += float64(v) * float64(v)
	}

	rms := math.Sqrt(sum / float64(len(a)))
	if rms == 0 {
		return math.Inf(-1)
	}

	return 20 * math.Log10(rms)
}

// HasNaNOrInf returns true if any sample is NaN or ±Inf.
func HasNaNOrInf(a []float32) bool {
	for _, v := range a {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}

	return false
}
