# whisper-turbo.c

Turn audio into text with timestamps and speaker labels — on a CPU, without a GPU.

Self-host Whisper large-v3-turbo through an HTTP API or command line. The native C
runtime requires no Python or C++ dependencies.

| Capability | What you get |
| --- | --- |
| Transcription | Speech-to-text with automatic language detection |
| Speaker labels | Timestamped segments showing who spoke when; optional, experimental diarization |
| Hardware | CPU-only inference, with acceleration for Intel and AMD x86 CPUs |
| Integration | HTTP uploads and JSON/text responses; a documented subset of the OpenAI transcription API |
| Input | Mono 16-bit PCM WAV at 16 kHz; up to 5 minutes per HTTP request |

Speaker labels distinguish voices, not real-world identities. Overlapping voices
are not separated. All model aliases run locally; audio is not sent to OpenAI.
See [API capabilities and limits](docs/http-api.md).

## Transcription and speaker-labeling costs

| Service | Hosting / hardware | Cost per audio hour |
| --- | --- | ---: |
| whisper-turbo.c | Self-hosted CPU; InstaCloud estimate | **~$0.136** |
| [WhisperX + pyannote via Whipscribe](https://whipscribe.com/pricing) | Hosted GPU; $24 / 5,000-minute pack | $0.288 |
| [OpenAI gpt-4o-transcribe-diarize](https://developers.openai.com/api/docs/pricing) | Hosted API | ~$0.360 |
| [AssemblyAI Universal-2 + diarization](https://www.assemblyai.com/pricing) | Hosted API | $0.170 |
| [AssemblyAI Universal-3.5 Pro + diarization](https://www.assemblyai.com/pricing) | Hosted API | $0.230 |

USD, checked September 11, 2026. Native cost is an active-processing estimate
from a short two-speaker test, assuming 8 fully utilized vCPUs and 1.1 GB RAM;
startup and idle costs are excluded. Hosted services use published prices.
Whipscribe includes speaker labels; $0.288/hour assumes full use of its
5,000-minute pack. It [runs WhisperX on GPUs](https://whipscribe.com/blog/whisperx-vs-whipscribe-2026);
this is a hosted-product price, not a self-hosted compute estimate.
Workloads and model quality differ. See [measurements and cost assumptions](docs/cost-comparison.md).

WhisperX also runs without a GPU, including alignment and speaker labeling.
Its CPU setting `--device cpu --compute_type int8` applies INT8 to transcription,
not to the separate diarization model.
See [WhisperX CPU usage](https://github.com/m-bain/whisperX/blob/v3.8.6/README.md#usage--command-line).

## Setup and compilation

Requires a C11 compiler, Make, POSIX APIs, libm, and zlib development headers.
For multithreaded x86 inference,
use a compiler with OpenMP support:

```sh
make server OPENMP=-fopenmp
OMP_NUM_THREADS=8 ./build/whisper-turbo-server turbo-q8.whtrbo 8080
```

The server binds to loopback by default. Set `WHISPER_API_KEY` before binding to a
non-loopback address, and use a TLS reverse proxy for remote access. See
[server configuration](docs/http-api.md#build-and-run).

To prepare the model and use the standalone CLI:

```sh
make x86-tools OPENMP=-fopenmp
./build/import-ggml ggml-large-v3-turbo.bin turbo-q8.whtrbo
OMP_NUM_THREADS=8 ./build/whisper-turbo-x86 turbo-q8.whtrbo speech.wav 448 fixed30
```

Weights are not included. The importer accepts the original GGML F16/F32
large-v3-turbo checkpoint, not a prequantized GGML model. The checkpoint SHA-256 is:

```text
1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69
```

The converted INT8 model is 847,660,800 bytes. The output path must not already
exist. See [model identities](docs/pins.json) for checkpoint details.

For the portable runtime without OpenMP:

```sh
make
./build/whisper-turbo-transcribe turbo-q8.whtrbo speech.wav 448 fixed30
```

### Tests

```sh
make check
```

Runs deterministic C checks for quantized kernels, attention parity, HTTP behavior,
and diarization. Diarization checks require zlib development headers and the library.

The root Dockerfile builds a benchmark image containing an external, pinned
whisper.cpp comparison executable. Its C++ toolchain is not required by the
native application.

## HTTP transcription

For a speaker-labeled transcript, start the server with
`WHISPER_DIARIZATION_MODELS=/path/to/community-1` and send:

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=gpt-4o-transcribe-diarize \
  -F response_format=diarized_json -F chunking_strategy=auto
```

Returns `task`, `duration`, combined `text`, timestamped `segments` with
`speaker` labels (`A`, `B`, …), and duration `usage`. Optional reference names and
2–10-second WAV data URLs can label matching speakers. `stream=true` returns SSE
delta, segment, and done events, buffered until inference completes.
See [diarized transcription](docs/http-api.md#diarized-transcription) for limits.

For transcription without diarization:

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=whisper-1 -F language=en
```

Returns `{"text":"..."}`. Add `-F response_format=text` for plain text. `whisper-1`
is an alias for the local Turbo model, not OpenAI's hosted model. The server keeps
one model loaded, accepts one transcription at a time, and processes WAV recordings
up to 300 seconds in bounded windows. See [API options and limits](docs/http-api.md).

## Audio and output

The combined CLI uses exactly the same inference and response code as the HTTP
diarization endpoint, including automatic language detection and the 300-second
limit:

```sh
OMP_NUM_THREADS=8 WHISPER_ACTIVATIONS=int8 ./build/whisper-turbo-diarize \
  turbo-q8.whtrbo /path/to/community-1 speech.wav en
```

It writes diarized JSON to stdout and pass counts/timing to stderr. Omit `en` for
automatic language detection. Build with `make diarized-cli OPENMP=-fopenmp`.

Input must be mono 16-bit PCM WAV at 16 kHz. The older ASR-only command-line transcriber processes only the
first 30 seconds, uses English greedy decoding, and prints text on a `TRANSCRIPT:`
line alongside timing logs. Language detection, timestamps, and long-audio
processing are not supported by that CLI. The HTTP server supports language
detection and recordings up to 300 seconds.

```text
whisper-turbo-x86 MODEL.whtrbo AUDIO.wav [MAX_TOKENS] [fixed30|compact]
```

`MAX_TOKENS` accepts 6–448 and includes decoder context; use 448 to reduce the
risk of truncating the transcript. `fixed30` uses the full 30-second encoder
window. `compact` uses a shorter window for short clips and can change results.

## CPU options

CPU features are detected automatically. The x86 runtime includes AVX2/FMA,
AVX-512, and VNNI kernels with a scalar fallback.

| Environment variable | Behavior |
| --- | --- |
| `OMP_NUM_THREADS=8` | Use eight threads in an OpenMP build. |
| `WHISPER_SIMD=scalar`, `avx2`, or `avx512` | Select a CPU path, subject to hardware support. |
| `WHISPER_ACTIVATIONS=int8` | Enable experimental INT8 encoder activations; decoder activations remain FP32. |
| `WHISPER_ATTENTION_PACK=0` | Disable attention K/V packing, which is enabled by default. |

INT8 weights are used with FP32 activations by default. INT8 encoder activations
have limited transcription-accuracy validation and are opt-in:

```sh
OMP_NUM_THREADS=8 WHISPER_ACTIVATIONS=int8 \
  ./build/whisper-turbo-x86 turbo-q8.whtrbo speech.wav 448 fixed30
```

## Benchmark details

LibriSpeech is a public collection of read English speech used to evaluate
transcription systems. **Word error rate (WER)** counts substituted, missing and
extra words relative to a reference transcript; lower is better. The small
subsets below are not representative of every language or recording condition.

### CPU performance

**September 11, 2026 — Intel Xeon 6975P-C, 8 shared vCPUs / 8 threads.**
Warm HTTP inference with one resident model.

| Workload | Audio duration | Processing time | Estimated $/audio hour |
| --- | ---: | ---: | ---: |
| ASR: 10 LibriSpeech clips | 91.525 s total | **92.605 s total** | **$0.240** |
| ASR + alignment + diarization: two-speaker fixture | 26 s | **14.876 s mean** | **$0.136** |

ASR: **1.1952% WER** on this 251-word subset. The combined fixture produces
timestamped text with A–B–A speaker labels. Combined times average two requests;
ASR times sum ten.

INT8 weights and encoder activations, opt-in INT8 decoder projections, AVX-512
diarization and batched LSTM input projections. **Diarization remains FP32.**
See [settings and results](docs/cpu-optimizations.md)
and [raw measurements](benchmarks/results/cpu-optimizations/measurements.json).

Costs assume **8 fully utilized vCPUs + 1.1 decimal GB RAM** during processing,
using [InstaCloud rates](https://instacloud.com/pricing) checked September 11, 2026.
Estimates cover active processing only, normalized by each workload's audio
duration; startup and idle costs are excluded. See
[cost methodology](docs/cost-comparison.md).

### Transcription accuracy comparison — 4 CPUs

**September 11, 2026 — InstaCloud, Intel Xeon 6975P-C, 4 CPUs / 4 threads.**
Transcription only, using Turbo INT8 weights. Both resident servers process the
same **50 test-clean utterances: 328.27 seconds, 915 reference words, one English
speaker**. Startup and one warm-up request per engine are excluded.

| Engine | WER ↓ | Substitutions / deletions / insertions | Mean request time |
| --- | ---: | ---: | ---: |
| whisper-turbo.c | **3.72%** | 25 / 9 / 0 | 21.21 s |
| whisper.cpp Q8_0 | **3.93%** | 26 / 10 / 0 | **11.99 s** |

All 100 measured requests succeeded. The native result has two fewer word errors;
whisper.cpp is 1.77× faster on this slice. **This single-speaker subset does not
establish a general accuracy ranking or full-corpus WER.** Both outputs use the
same Unicode normalization, but the engines have different decoding policies.
See [results and exact configurations](benchmarks/results/librispeech-50/README.md),
[the C benchmark](benchmarks/wer/README.md), and
[published whisper.cpp comparisons](benchmarks/wer/README.md#dataset-and-published-results).

### Inference architecture

Run Whisper large-v3-turbo speech recognition on CPU with a native C11 runtime,
INT8 model weights, and AVX2/AVX-512 acceleration. Model conversion and inference
require no Python or C++ dependencies.

The application provides a command-line transcriber and a [C HTTP server](docs/http-api.md)
with `POST /v1/audio/transcriptions`, multipart uploads, and JSON/text responses.
The endpoint supports a documented subset of the OpenAI transcription interface,
including `gpt-4o-transcribe-diarize`-shaped speaker-segment responses. All accepted
model names select local C inference, not OpenAI-hosted weights.

The [experimental C-only Community-1 diarizer](docs/diarization.md) can run
standalone or in the combined CLI/HTTP transcription pipeline. Each audio window
is encoded once; [quality-gated decoder retries](docs/http-api.md#decoding-and-recovery)
reuse that output. Community-1 diarizes the recording once. Cross-attention alignment assigns
the existing text to speakers without retranscribing turns. Overlapping voices
are not separated.

## License

MIT, with Apache-2.0 components in the optional diarizer. See [LICENSE](LICENSE)
and [diarization attribution](THIRD_PARTY_DIARIZATION.md), plus
[alignment attribution](THIRD_PARTY.md). Based on the Whisper Turbo implementation in
[llm-in-c](https://github.com/baryhuang/llm-in-c).
