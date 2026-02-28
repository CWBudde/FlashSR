# FlashSR Web Demo

This demo runs a Go/WASM kernel in the browser and checks WAV quality based on
sample rate:

- If input sample rate is below `48 kHz`, it resamples to `48 kHz` using
  `resample.NewPolyphase(...)`.
- If input sample rate is `48 kHz` or higher, it keeps the input unchanged.

## Local run

```bash
./web/build-wasm.sh
python3 -m http.server 8080 -d web/dist
```

Open <http://localhost:8080>.

## GitHub Pages

`/.github/workflows/deploy-pages.yml` builds the wasm kernel, copies
`wasm_exec.js` and web assets into `web/dist`, then deploys to GitHub Pages.
