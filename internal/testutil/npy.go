package testutil

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// LoadNPYFloat32 loads a 1-D numpy float32 array saved with np.save.
// Only supports NPY format v1.0 / v2.0 with dtype '<f4' (little-endian float32).
func LoadNPYFloat32(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("npy: open %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	return readNPYFloat32(f)
}

func readNPYFloat32(r io.Reader) ([]float32, error) {
	// Magic: \x93NUMPY
	magic := make([]byte, 6)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("npy: read magic: %w", err)
	}

	if string(magic) != "\x93NUMPY" {
		return nil, errors.New("npy: not a numpy file")
	}

	// Version: major, minor (1 byte each).
	var ver [2]byte
	if _, err := io.ReadFull(r, ver[:]); err != nil {
		return nil, fmt.Errorf("npy: read version: %w", err)
	}

	// Header length: 2 bytes (v1) or 4 bytes (v2), little-endian.
	var headerLen uint32
	switch ver[0] {
	case 1:
		var hl [2]byte
		if _, err := io.ReadFull(r, hl[:]); err != nil {
			return nil, fmt.Errorf("npy: read header len: %w", err)
		}

		headerLen = uint32(binary.LittleEndian.Uint16(hl[:]))
	case 2:
		var hl [4]byte
		if _, err := io.ReadFull(r, hl[:]); err != nil {
			return nil, fmt.Errorf("npy: read header len v2: %w", err)
		}

		headerLen = binary.LittleEndian.Uint32(hl[:])
	default:
		return nil, fmt.Errorf("npy: unsupported version %d.%d", ver[0], ver[1])
	}

	hdr := make([]byte, headerLen)
	if _, err := io.ReadFull(r, hdr); err != nil {
		return nil, fmt.Errorf("npy: read header: %w", err)
	}

	hdrStr := strings.TrimSpace(string(hdr))

	dtype, err := extractString(hdrStr, "descr")
	if err != nil {
		return nil, fmt.Errorf("npy: parse dtype: %w", err)
	}

	if dtype != "<f4" && dtype != "float32" {
		return nil, fmt.Errorf("npy: unsupported dtype %q (want '<f4')", dtype)
	}

	n, err := extractTotalElements(hdrStr)
	if err != nil {
		return nil, fmt.Errorf("npy: parse shape: %w", err)
	}

	data := make([]float32, n)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return nil, fmt.Errorf("npy: read data: %w", err)
	}

	return data, nil
}

// extractString parses a Python dict literal for a string value by key.
func extractString(hdr, key string) (string, error) {
	re := regexp.MustCompile(`'` + key + `'\s*:\s*'([^']*)'`)
	m := re.FindStringSubmatch(hdr)

	if m == nil {
		return "", fmt.Errorf("key %q not found", key)
	}

	return m[1], nil
}

// extractTotalElements parses the shape tuple and returns the product of all dims.
func extractTotalElements(hdr string) (int, error) {
	re := regexp.MustCompile(`'shape'\s*:\s*\(([^)]*)\)`)
	m := re.FindStringSubmatch(hdr)

	if m == nil {
		return 0, errors.New("'shape' not found")
	}

	parts := strings.Split(m[1], ",")
	total := 1

	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}

		v, err := strconv.Atoi(p)
		if err != nil {
			return 0, fmt.Errorf("bad dimension %q: %w", p, err)
		}

		total *= v
	}

	return total, nil
}
