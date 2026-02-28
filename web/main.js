const fileInput = document.getElementById("wav-file");
const qualitySelect = document.getElementById("quality");
const runButton = document.getElementById("run");
const downloadButton = document.getElementById("download");
const kernelStatus = document.getElementById("kernel-status");
const analysisStatus = document.getElementById("analysis-status");
const resultStatus = document.getElementById("result-status");
const inputPlayer = document.getElementById("input-player");
const outputPlayer = document.getElementById("output-player");

const state = {
  kernel: null,
  selectedFile: null,
  selectedBytes: null,
  outputBytes: null,
  wasResampled: false,
  outputURL: "",
  inputURL: "",
  working: false,
};

function formatError(err) {
  if (!err) return "unknown error";
  if (typeof err === "string") return err;
  if (err.message) return err.message;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

function setKernelStatus(text) {
  kernelStatus.textContent = text;
}

function setAnalysisStatus(text) {
  analysisStatus.textContent = text;
}

function setResultStatus(text) {
  resultStatus.textContent = text;
}

function setBusy(isBusy) {
  state.working = isBusy;
  runButton.disabled = isBusy || !state.kernel || !state.selectedFile;
  qualitySelect.disabled = isBusy;
  fileInput.disabled = isBusy;
}

function setInputPreview(file) {
  if (state.inputURL) {
    URL.revokeObjectURL(state.inputURL);
    state.inputURL = "";
  }
  state.inputURL = URL.createObjectURL(file);
  inputPlayer.src = state.inputURL;
}

function setOutputPreview(outputBytes) {
  if (state.outputURL) {
    URL.revokeObjectURL(state.outputURL);
    state.outputURL = "";
  }
  const blob = new Blob([outputBytes], { type: "audio/wav" });
  state.outputURL = URL.createObjectURL(blob);
  outputPlayer.src = state.outputURL;
}

function makeOutputFilename(inputName, wasResampled) {
  const dot = inputName.lastIndexOf(".");
  const base = dot >= 0 ? inputName.slice(0, dot) : inputName;
  if (wasResampled) {
    return `${base}_48k.wav`;
  }
  return `${base}_checked.wav`;
}

async function instantiateKernel() {
  if (typeof Go === "undefined") {
    throw new Error("Go runtime was not loaded (missing wasm_exec.js)");
  }

  const go = new Go();
  const wasmURL = "./flashsr-kernel.wasm";

  let instance;
  try {
    const result = await WebAssembly.instantiateStreaming(fetch(wasmURL), go.importObject);
    instance = result.instance;
  } catch {
    const response = await fetch(wasmURL);
    if (!response.ok) {
      throw new Error(`fetch ${wasmURL} failed (${response.status})`);
    }
    const bytes = await response.arrayBuffer();
    const result = await WebAssembly.instantiate(bytes, go.importObject);
    instance = result.instance;
  }

  go.run(instance);
  const kernel = globalThis.FlashSRWebKernel;
  if (!kernel) {
    throw new Error("FlashSRWebKernel was not found after wasm startup");
  }
  if (typeof kernel.analyzeWAV !== "function" || typeof kernel.processWAV !== "function") {
    throw new Error("FlashSRWebKernel is missing required APIs");
  }

  return kernel;
}

async function readSelectedFileBytes() {
  if (!state.selectedFile) {
    throw new Error("no WAV file selected");
  }

  if (state.selectedBytes) {
    return state.selectedBytes;
  }

  const buf = await state.selectedFile.arrayBuffer();
  state.selectedBytes = new Uint8Array(buf);
  return state.selectedBytes;
}

function renderAnalysis(meta) {
  const lines = [
    `Input sample rate: ${meta.inputSampleRate} Hz`,
    `Target sample rate: ${meta.targetSampleRate} Hz`,
    `Channels: ${meta.inputChannels}`,
    `Bit depth: ${meta.inputBitDepth}`,
    `Quality: ${meta.quality}`,
    `Needs resample: ${meta.needsResample ? "yes" : "no"}`,
    `${meta.message}`,
  ];
  setAnalysisStatus(lines.join("\n"));
}

function renderResult(result) {
  const lines = [
    `Input sample rate: ${result.inputSampleRate} Hz`,
    `Output sample rate: ${result.outputSampleRate} Hz`,
    `Resampled: ${result.wasResampled ? "yes" : "no"}`,
    `Resample mode: ${result.resampleMode}`,
    `Channels: ${result.inputChannels}`,
    `Quality: ${result.quality}`,
    `${result.message}`,
  ];

  if (typeof result.inputSamples === "number") {
    lines.push(`Input samples (mono): ${result.inputSamples}`);
  }
  if (typeof result.outputSamples === "number") {
    lines.push(`Output samples: ${result.outputSamples}`);
  }

  setResultStatus(lines.join("\n"));
}

async function analyzeSelectedWAV() {
  if (!state.kernel || !state.selectedFile) {
    return;
  }

  try {
    const bytes = await readSelectedFileBytes();
    const meta = await state.kernel.analyzeWAV(bytes);
    renderAnalysis(meta);
  } catch (err) {
    setAnalysisStatus(`Analyze failed:\n${formatError(err)}`);
  }
}

async function processSelectedWAV() {
  if (!state.kernel || !state.selectedFile || state.working) {
    return;
  }

  setBusy(true);
  downloadButton.disabled = true;
  setResultStatus("Running Go wasm kernel...");

  try {
    const bytes = await readSelectedFileBytes();
    const quality = String(qualitySelect.value || "balanced");
    const result = await state.kernel.processWAV(bytes, quality);

    renderResult(result);
    state.outputBytes = result.wavBytes;
    state.wasResampled = Boolean(result.wasResampled);
    setOutputPreview(state.outputBytes);
    downloadButton.disabled = false;
  } catch (err) {
    setResultStatus(`Processing failed:\n${formatError(err)}`);
    state.outputBytes = null;
    state.wasResampled = false;
    downloadButton.disabled = true;
  } finally {
    setBusy(false);
  }
}

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0] || null;
  state.selectedFile = file;
  state.selectedBytes = null;
  state.outputBytes = null;
  state.wasResampled = false;
  outputPlayer.removeAttribute("src");
  outputPlayer.load();
  downloadButton.disabled = true;

  if (!file) {
    setAnalysisStatus("No file selected.");
    runButton.disabled = true;
    return;
  }

  setInputPreview(file);
  runButton.disabled = !state.kernel;
  setResultStatus("No processing run yet.");
  setAnalysisStatus("Inspecting WAV metadata...");
  await analyzeSelectedWAV();
});

runButton.addEventListener("click", async () => {
  await processSelectedWAV();
});

downloadButton.addEventListener("click", () => {
  if (!state.outputBytes || !state.selectedFile) {
    return;
  }
  const blob = new Blob([state.outputBytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = makeOutputFilename(state.selectedFile.name, state.wasResampled);
  a.click();
  URL.revokeObjectURL(url);
});

async function boot() {
  setBusy(true);
  setKernelStatus("Loading wasm kernel...");

  try {
    const kernel = await instantiateKernel();
    state.kernel = kernel;
    const version = String(kernel.version || "unknown");
    const target = Number(kernel.targetSampleRate || 0);
    setKernelStatus(`ready\nversion: ${version}\ntarget: ${target} Hz`);
  } catch (err) {
    setKernelStatus(`failed\n${formatError(err)}`);
    setAnalysisStatus("Kernel failed to initialize.");
    setResultStatus("Kernel failed to initialize.");
    return;
  } finally {
    setBusy(false);
  }

  if (state.selectedFile) {
    await analyzeSelectedWAV();
  }
}

void boot();
