// Polyphase FIR resampler — stateful, streaming-friendly.
//
// Adapted from github.com/cwbudde/algo-dsp (MIT License).
package resample

import "math"

// Quality controls anti-aliasing filter aggressiveness.
type Quality int

const (
	// QualityFast prioritises lower CPU (16 taps/phase, ~55 dB stopband).
	QualityFast Quality = iota
	// QualityBalanced is the default quality/performance trade-off (32 taps/phase, ~75 dB).
	QualityBalanced
	// QualityBest maximises stopband attenuation and passband flatness (64 taps/phase, ~90 dB).
	QualityBest
)

// Profile exposes the filter parameters for a given quality mode.
type Profile struct {
	TapsPerPhase      int
	CutoffScale       float64
	KaiserBeta        float64
	NominalStopbandDB float64
}

// QualityProfile returns the default filter profile for quality mode q.
func QualityProfile(q Quality) Profile {
	switch q {
	case QualityFast:
		return Profile{TapsPerPhase: 16, CutoffScale: 0.88, KaiserBeta: 5.0, NominalStopbandDB: 55}
	case QualityBest:
		return Profile{TapsPerPhase: 64, CutoffScale: 0.96, KaiserBeta: 9.0, NominalStopbandDB: 90}
	default:
		return Profile{TapsPerPhase: 32, CutoffScale: 0.92, KaiserBeta: 7.5, NominalStopbandDB: 75}
	}
}

// Option configures the polyphase resampler.
type Option func(*config)

// WithQuality selects a predefined anti-aliasing quality mode.
func WithQuality(q Quality) Option {
	return func(cfg *config) { cfg.quality = q }
}

// WithTapsPerPhase overrides the taps per polyphase branch.
func WithTapsPerPhase(n int) Option {
	return func(cfg *config) {
		if n > 0 {
			cfg.tapsPerPhase = n
		}
	}
}

// WithCutoffScale overrides the normalised cutoff in (0, 1].
func WithCutoffScale(v float64) Option {
	return func(cfg *config) {
		if v > 0 && v <= 1 {
			cfg.cutoffScale = v
		}
	}
}

// WithKaiserBeta overrides the Kaiser window beta parameter.
func WithKaiserBeta(beta float64) Option {
	return func(cfg *config) {
		if beta >= 0 {
			cfg.kaiserBeta = beta
		}
	}
}

// WithMaxDenominator caps the denominator used in rational approximation.
func WithMaxDenominator(n int) Option {
	return func(cfg *config) {
		if n > 0 {
			cfg.maxDen = n
		}
	}
}

type config struct {
	quality      Quality
	tapsPerPhase int
	cutoffScale  float64
	kaiserBeta   float64
	maxDen       int
}

func defaultConfig() config {
	return config{quality: QualityBalanced, maxDen: 4096}
}

func (c config) finalized() config {
	p := QualityProfile(c.quality)

	if c.tapsPerPhase <= 0 {
		c.tapsPerPhase = p.TapsPerPhase
	}

	if c.cutoffScale <= 0 || c.cutoffScale > 1 {
		c.cutoffScale = p.CutoffScale
	}

	if c.kaiserBeta <= 0 {
		c.kaiserBeta = p.KaiserBeta
	}

	if c.maxDen <= 0 {
		c.maxDen = 4096
	}

	return c
}

// polyphaseEngine is the internal float64 FIR engine (adapted from algo-dsp).
type polyphaseEngine struct {
	up, down   int
	phases     [][]float64
	maxPhaseLn int

	phase      int
	inputIndex int
	totalIn    int
	history    []float64
}

func newPolyphaseEngine(up, down int, cfg config) (*polyphaseEngine, error) {
	_, phases, maxPhaseLn, err := designPolyphaseFIR(up, down, cfg)
	if err != nil {
		return nil, err
	}

	return &polyphaseEngine{
		up:         up,
		down:       down,
		phases:     phases,
		maxPhaseLn: maxPhaseLn,
		history:    make([]float64, 0, max(0, maxPhaseLn-1)),
	}, nil
}

func (e *polyphaseEngine) reset() {
	e.phase = 0
	e.inputIndex = 0
	e.totalIn = 0
	e.history = e.history[:0]
}

func (e *polyphaseEngine) process(input []float64) []float64 {
	if len(input) == 0 {
		return nil
	}

	nOut := e.predictOutputLen(len(input))
	out := make([]float64, 0, nOut)

	work := make([]float64, len(e.history)+len(input))
	copy(work, e.history)
	copy(work[len(e.history):], input)

	baseIndex := e.totalIn - len(e.history)
	lastAvail := e.totalIn + len(input) - 1

	for e.inputIndex <= lastAvail {
		taps := e.phases[e.phase]

		var y float64

		for k, c := range taps {
			idx := e.inputIndex - k
			if idx < baseIndex || idx > lastAvail {
				continue
			}

			y += c * work[idx-baseIndex]
		}

		out = append(out, y)

		e.phase += e.down
		e.inputIndex += e.phase / e.up
		e.phase %= e.up
	}

	e.totalIn += len(input)

	keep := min(max(0, e.maxPhaseLn-1), len(work))
	e.history = append(e.history[:0], work[len(work)-keep:]...)

	return out
}

func (e *polyphaseEngine) predictOutputLen(inputLen int) int {
	if inputLen <= 0 {
		return 0
	}

	lastAvail := e.totalIn + inputLen - 1
	i := e.inputIndex
	phase := e.phase
	count := 0

	for i <= lastAvail {
		count++
		phase += e.down
		i += phase / e.up
		phase %= e.up
	}

	return count
}

// polyphaseResampler wraps polyphaseEngine and satisfies the Resampler interface
// with float32 input/output.
type polyphaseResampler struct {
	eng *polyphaseEngine
}

// newPolyphaseResampler creates a polyphase resampler for the given integer rate pair.
func newPolyphaseResampler(inRate, outRate int, opts []Option) (Resampler, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	cfg = cfg.finalized()

	up, down := approximateRatio(float64(outRate)/float64(inRate), cfg.maxDen)

	eng, err := newPolyphaseEngine(up, down, cfg)
	if err != nil {
		return nil, err
	}

	return &polyphaseResampler{eng: eng}, nil
}

func (r *polyphaseResampler) Process(in []float32) ([]float32, error) {
	f64 := make([]float64, len(in))
	for i, v := range in {
		f64[i] = float64(v)
	}

	out64 := r.eng.process(f64)

	out := make([]float32, len(out64))
	for i, v := range out64 {
		out[i] = float32(math.Max(-1, math.Min(1, v))) // clamp to [-1,1]
	}

	return out, nil
}

func (r *polyphaseResampler) Reset() {
	r.eng.reset()
}
