// Package testutil provides signal generators and comparison helpers for tests.
package testutil

import (
	"math"
	"math/rand"
)

// Sine generates n float32 samples of a pure sine wave.
func Sine(freq, sampleRate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(2 * math.Pi * float64(freq) * float64(i) / float64(sampleRate)))
	}

	return out
}

// SineSweep generates n samples of a linear-frequency sine sweep from freqStart
// to freqEnd Hz using the instantaneous-phase method (no discontinuities).
func SineSweep(freqStart, freqEnd float64, sampleRate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		t := float64(i) / float64(sampleRate)
		T := float64(n) / float64(sampleRate)
		// Instantaneous frequency increases linearly; integrate to get phase.
		phase := 2 * math.Pi * t * (freqStart + (freqEnd-freqStart)*t/(2*T))
		out[i] = float32(math.Sin(phase))
	}

	return out
}

// PinkNoise generates n samples of approximate pink noise (−3 dB/octave power
// spectrum) using a simple Voss-McCartney filter bank.
func PinkNoise(seed int64, n int) []float32 {
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic, not security-sensitive

	const numRows = 16
	rows := make([]float64, numRows)
	running := 0.0

	out := make([]float32, n)
	for i := range out {
		// Find lowest set bit of (i+1) to determine which row to update.
		k := i + 1
		bit := k & (-k) // lowest set bit

		for b := 0; b < numRows; b++ {
			if bit>>b&1 == 1 {
				running -= rows[b]
				rows[b] = rng.Float64()*2 - 1
				running += rows[b]

				break
			}
		}

		out[i] = float32(running / float64(numRows))
	}

	// Normalise to [-1, 1].
	return normalise(out)
}

// PeakAbs returns the absolute peak value of a.
func PeakAbs(a []float32) float32 {
	peak := float32(0)
	for _, v := range a {
		if v < 0 {
			v = -v
		}

		if v > peak {
			peak = v
		}
	}

	return peak
}

func normalise(a []float32) []float32 {
	peak := PeakAbs(a)
	if peak == 0 {
		return a
	}

	for i := range a {
		a[i] /= peak
	}

	return a
}
