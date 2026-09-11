# Community-1 speaker diarization in C

An experimental, CPU-only port of
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
It produces timestamped speaker segments and an exclusive timeline with no
simultaneous speakers. The executable uses C11, libm, POSIX APIs, and zlib—no
Python, PyTorch, ONNX Runtime, or C++ runtime dependency.

This executable is a standalone diarizer. The same C pipeline is also available
inside the [diarized transcription HTTP endpoint](http-api.md#diarized-transcription),
which combines one ASR pass with one diarization pass and alignment-based speaker
assignment. The standalone executable does not transcribe audio, assign speakers to
Whisper words, or provide an HTTP API.

## Build

```sh
make diarization
```

For a compiler with OpenMP support:

```sh
make diarization OPENMP=-fopenmp
```

Apple Silicon uses NEON dot products. x86-64 uses AVX2/FMA when available and a
scalar fallback otherwise. OpenMP parallelizes convolutions. Use
`OMP_NUM_THREADS=8` to bound the thread count in an OpenMP build.

## Model files

Accept the model's access conditions on Hugging Face and download the following
files from revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`. Store them in a
local model directory using these filenames:

| Upstream file | Local filename |
| --- | --- |
| `segmentation/pytorch_model.bin` | `segmentation-pytorch_model.bin` |
| `embedding/pytorch_model.bin` | `embedding-pytorch_model.bin` |
| `plda/xvec_transform.npz` | `plda-xvec_transform.npz` |
| `plda/plda.npz` | `plda-plda.npz` |

The original files are read directly; no conversion or Python installation is
required. The reader handles the pinned uncompressed ZIP checkpoints and a
bounded subset of pickle metadata. Serialized globals are never imported or
executed. ZIP checksums, tensor bounds, strides, dtypes, and graph shapes are
validated. This is not a general-purpose PyTorch checkpoint loader.

Model checksums:

```text
7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e  segmentation-pytorch_model.bin
6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929  embedding-pytorch_model.bin
325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f  plda-xvec_transform.npz
9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255  plda-plda.npz
```

Weights are separate from the application license. See
[attribution and model licensing](../THIRD_PARTY_DIARIZATION.md).

## Run

```sh
OMP_NUM_THREADS=8 ./build/diarize-community /path/to/models conversation.wav
```

Input must be mono PCM16 WAV at 16 kHz, between one sample and 300 seconds.
Unsupported formats and longer recordings are rejected, not silently truncated.
Very short speech may not provide enough clean audio for speaker clustering.

JSON is written to stdout; progress and timing are written to stderr. The output
contains:

- `audio_seconds`: recording duration.
- `num_speakers`: number of speakers present in the output.
- `diarization`: segments with `start`, `end`, and `speaker`; overlaps are allowed.
- `exclusive_diarization`: segments with at most one speaker at a time.

Speaker labels are anonymous, starting with `SPEAKER_00` in order of first
appearance. They are local to each recording, not persistent identities.
Segment times are in seconds relative to the input recording. The JSON includes
`"implementation":"experimental-c"` to distinguish this port from upstream.

## Implementation and limits

The pipeline implements ten-second segmentation windows with one-second hops,
SincNet, four bidirectional LSTM layers, seven-class powerset decoding, Kaldi-style
80-bin filterbanks, WeSpeaker ResNet34, overlap-aware weighted statistics pooling,
PLDA, centroid-linkage initialization, and VBx clustering. Community-1 clustering
parameters are `threshold=0.6`, `Fa=0.07`, and `Fb=0.8`.

Workspaces are bounded to one neural-network window; clustering is limited to
876 training embeddings and 32 output speakers. The CLI does not support forced
speaker counts, enrollment, streaming, resampling, or stereo downmixing.
Missing embeddings in a local window are omitted from speaker voting; overlapping
windows can supply the evidence. Insufficient clean speech across the recording
produces an error instead of invented speaker labels.
Detected silence produces empty timelines.

The port is not certified as numerically equivalent to pyannote.audio. It uses
batch size one, shares ResNet features across local speakers, clears segmentation
masks beyond the recording boundary, and uses the supplied PLDA diagonalization
directly. Floating-point reductions and tied cluster assignments can differ from
upstream. No upstream parity corpus or standard diarization-error-rate evaluation
has been completed. Overlapping-speaker accuracy is unverified.

## Validation and performance

On InstaCloud's Intel Xeon 6975P-C with eight OpenMP threads, standalone
diarization of a 27.27-second recording took **8.60 seconds median** across three
cold-model runs. Maximum observed process RSS was **70.4 MB**, and cgroup memory
was **78.4 MB** (decimal). Sequential native transcription plus diarization took
**22.54 seconds median**, with a maximum cgroup peak of **955.7 MB**. These are
separate CLI processes, not a resident HTTP service. See
[the cloud comparison and methodology](instacloud-diarization.md).

An assembled 26-second, two-voice A–B–A recording produced two speaker labels and
preserved speaker A's identity after speaker B on both Xeon and Apple Silicon.
The Xeon run took 7.38 seconds at eight threads and passed the C identity check.
On Apple M3 Pro, a single-threaded native run took 69.74 seconds and reached
171,393,024 bytes peak process RSS (decimal 171.4 MB). These use different CPUs
and thread counts and are smoke tests, not a controlled cross-platform speed
comparison or general accuracy benchmark.

Concurrently resident Whisper-plus-diarization memory and HTTP-service memory
remain unmeasured.

Run model-independent C checks:

```sh
make check-diarization
```

Validate the downloaded checkpoints, segmentation output, and PLDA path:

```sh
./build/diarization-test /path/to/models
```

The C unit checks and a real-audio end-to-end run pass AddressSanitizer and
UndefinedBehaviorSanitizer on Apple Silicon. The checks cover dot products,
powerset mapping, filterbank centering, VBx normalization and sign invariance,
invalid archives, corrupted tensor payloads, silence, checkpoint tensor views,
and identical-embedding clustering.

Build an A–B–A identity fixture from two known, distinct voices. Each input must
contain at least eight seconds of mono 16 kHz PCM16 audio:

```sh
make build/diarization-fixture
./build/diarization-fixture speaker-a.wav speaker-b.wav aba.wav
./build/diarize-community /path/to/models aba.wav > aba.json
./build/diarization-test --check-aba aba.json
```

The identity check verifies consistent labels for both A sections and a different
label for B. It is not a diarization-error-rate calculation.
