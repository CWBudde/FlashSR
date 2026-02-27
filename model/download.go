package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	// DefaultRepo is the Hugging Face repository hosting the FlashSR model.
	// Note: hance-ai/FlashSR is gated; YatharthS/FlashSR is the public mirror.
	DefaultRepo = "YatharthS/FlashSR"
	// DefaultRevision is the Git ref used when downloading from Hugging Face.
	DefaultRevision = "main"
	// DefaultFilename is the model file path within the repository.
	DefaultFilename = "onnx/model.onnx"
)

// DownloadOptions controls how the model is fetched from Hugging Face.
type DownloadOptions struct {
	// Repo is the Hugging Face repository (default: DefaultRepo).
	Repo string
	// Revision is the branch/commit/tag (default: DefaultRevision).
	Revision string
	// Filename is the file path within the repo (default: DefaultFilename).
	Filename string
	// OutPath is the local destination. Directories are created as needed.
	OutPath string
	// HFToken is an optional Hugging Face API token (Bearer auth).
	HFToken string
	// Stdout receives progress messages. Defaults to io.Discard.
	Stdout io.Writer
}

func (o *DownloadOptions) fill() {
	if o.Repo == "" {
		o.Repo = DefaultRepo
	}

	if o.Revision == "" {
		o.Revision = DefaultRevision
	}

	if o.Filename == "" {
		o.Filename = DefaultFilename
	}

	if o.Stdout == nil {
		o.Stdout = io.Discard
	}
}

// AccessDeniedError is returned when the server responds with 401 or 403.
type AccessDeniedError struct {
	Repo string
	Msg  string
}

func (e *AccessDeniedError) Error() string {
	if e.Msg != "" {
		return e.Msg
	}

	return "access denied for " + e.Repo
}

// DownloadResult holds the outcome of a successful Download call.
type DownloadResult struct {
	// Path is the absolute path of the saved file.
	Path string
	// SHA256 is the lowercase hex SHA256 of the file as downloaded.
	SHA256 string
	// Skipped is true when the file was already present with the correct hash.
	Skipped bool
}

// Download fetches the FlashSR ONNX model from Hugging Face and saves it to
// opts.OutPath. If a file already exists at that path and its SHA256 matches
// the server's ETag, the download is skipped.
//
// Returns the SHA256 of the file on disk (whether freshly downloaded or
// already present) so callers can pin the value in ExpectedSHA256.
func Download(opts DownloadOptions) (DownloadResult, error) {
	opts.fill()

	if opts.OutPath == "" {
		return DownloadResult{}, errors.New("model: OutPath is required")
	}

	absPath, err := filepath.Abs(opts.OutPath)
	if err != nil {
		return DownloadResult{}, fmt.Errorf("model: resolve out path: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
		return DownloadResult{}, fmt.Errorf("model: create output directory: %w", err)
	}

	url := fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s",
		opts.Repo, opts.Revision, opts.Filename)

	client := &http.Client{Timeout: 0}

	// Check existing file against server ETag before downloading.
	if existing, ok := existingFileHash(absPath); ok {
		serverHash, err := resolveServerHash(client, url, opts.HFToken)
		if err == nil && serverHash != "" && existing == serverHash {
			_, _ = fmt.Fprintf(opts.Stdout, "skip download: %s (sha256 match)\n", opts.Filename)
			return DownloadResult{Path: absPath, SHA256: existing, Skipped: true}, nil
		}
	}

	_, _ = fmt.Fprintf(opts.Stdout, "downloading %s from %s\n", opts.Filename, opts.Repo)

	actual, err := downloadToFile(client, url, opts.HFToken, absPath, opts.Stdout)
	if err != nil {
		return DownloadResult{}, err
	}

	_, _ = fmt.Fprintf(opts.Stdout, "saved %s (sha256=%s)\n", absPath, actual)

	return DownloadResult{Path: absPath, SHA256: actual}, nil
}

// downloadToFile streams url to outPath via a .tmp sibling, returns sha256.
func downloadToFile(client *http.Client, url, token, outPath string, stdout io.Writer) (string, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("model: build request: %w", err)
	}

	setAuthHeader(req, token)

	resp, err := client.Do(req) //nolint:bodyclose // closed below
	if err != nil {
		return "", fmt.Errorf("model: download request failed: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return "", &AccessDeniedError{
			Msg: fmt.Sprintf("access denied (%s); set HF_TOKEN or --hf-token", resp.Status),
		}
	}

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return "", fmt.Errorf("model: download failed: %s", resp.Status)
	}

	tmp := outPath + ".tmp"

	fh, err := os.Create(tmp)
	if err != nil {
		return "", fmt.Errorf("model: create temp file: %w", err)
	}

	h := sha256.New()
	mw := io.MultiWriter(fh, h)

	var written int64
	total := resp.ContentLength
	buf := make([]byte, 64*1024)
	lastPrint := time.Now()

	for {
		n, readErr := resp.Body.Read(buf)

		if n > 0 {
			wn, writeErr := mw.Write(buf[:n])
			if writeErr != nil {
				_ = fh.Close()
				_ = os.Remove(tmp)

				return "", fmt.Errorf("model: write temp file: %w", writeErr)
			}

			written += int64(wn)

			if time.Since(lastPrint) > 700*time.Millisecond {
				if total > 0 {
					pct := float64(written) * 100 / float64(total)
					_, _ = fmt.Fprintf(stdout, "  %.1f%% (%d / %d bytes)\n", pct, written, total)
				} else {
					_, _ = fmt.Fprintf(stdout, "  %d bytes\n", written)
				}

				lastPrint = time.Now()
			}
		}

		if readErr == io.EOF {
			break
		}

		if readErr != nil {
			_ = fh.Close()
			_ = os.Remove(tmp)

			return "", fmt.Errorf("model: read during download: %w", readErr)
		}
	}

	if err := fh.Close(); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("model: close temp file: %w", err)
	}

	if err := os.Rename(tmp, outPath); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("model: rename temp file: %w", err)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// resolveServerHash issues a HEAD request and extracts the SHA256 from ETag
// headers that Hugging Face LFS populates (X-Linked-Etag, Etag).
func resolveServerHash(client *http.Client, url, token string) (string, error) {
	req, err := http.NewRequest(http.MethodHead, url, nil)
	if err != nil {
		return "", err
	}

	setAuthHeader(req, token)

	resp, err := client.Do(req) //nolint:bodyclose
	if err != nil {
		return "", err
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode > 399 {
		return "", fmt.Errorf("HEAD %s: %s", url, resp.Status)
	}

	for _, key := range []string{"X-Linked-Etag", "Etag"} {
		v := normalizeETag(resp.Header.Get(key))
		if isSHA256Hex(v) {
			return v, nil
		}
	}

	return "", nil // no usable hash in headers — caller should download
}

func existingFileHash(path string) (string, bool) {
	f, err := os.Open(path)
	if err != nil {
		return "", false
	}

	defer func() { _ = f.Close() }()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", false
	}

	return hex.EncodeToString(h.Sum(nil)), true
}

func setAuthHeader(req *http.Request, token string) {
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

func normalizeETag(v string) string {
	for _, trim := range []string{`"`, "W/"} {
		v = trimAll(v, trim)
	}

	return v
}

func trimAll(s, cut string) string {
	for {
		prev := s
		s = trimEdges(s, cut)

		if s == prev {
			return s
		}
	}
}

func trimEdges(s, cut string) string {
	if len(s) >= len(cut) && s[:len(cut)] == cut {
		s = s[len(cut):]
	}

	if len(s) >= len(cut) && s[len(s)-len(cut):] == cut {
		s = s[:len(s)-len(cut)]
	}

	return s
}

func isSHA256Hex(v string) bool {
	if len(v) != 64 {
		return false
	}

	for _, c := range v {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}

	return true
}
