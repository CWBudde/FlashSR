package ort

import (
	"testing"
)

// TestORT_ShapeDetection verifies that inputShape and outputShape build the
// correct tensor dimensions for rank-2 ([1,N]) and rank-3 ([1,1,N]) models.
// This is a unit test that does not require ORT to be loaded.
func TestORT_ShapeDetection(t *testing.T) {
	cases := []struct {
		name       string
		inputRank  int
		outputRank int
		n          int64
		wantIn     []int64
		wantOut    []int64
	}{
		{
			name:       "rank2",
			inputRank:  2,
			outputRank: 2,
			n:          4000,
			wantIn:     []int64{1, 4000},
			wantOut:    []int64{1, 12000},
		},
		{
			name:       "rank3",
			inputRank:  3,
			outputRank: 3,
			n:          4000,
			wantIn:     []int64{1, 1, 4000},
			wantOut:    []int64{1, 1, 12000},
		},
		{
			name:       "rank3_default",
			inputRank:  0, // 0 triggers default branch → rank 3
			outputRank: 0,
			n:          8000,
			wantIn:     []int64{1, 1, 8000},
			wantOut:    []int64{1, 1, 24000},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := &Engine{
				inputRank:     tc.inputRank,
				outputRank:    tc.outputRank,
				upsampleRatio: 3,
			}

			got := []int64(e.inputShape(tc.n))
			if !eqSlice(got, tc.wantIn) {
				t.Errorf("inputShape(%d): got %v, want %v", tc.n, got, tc.wantIn)
			}

			outN := tc.n * int64(e.upsampleRatio)
			gotOut := []int64(e.outputShape(outN))
			if !eqSlice(gotOut, tc.wantOut) {
				t.Errorf("outputShape(%d): got %v, want %v", outN, gotOut, tc.wantOut)
			}
		})
	}
}

func eqSlice(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
