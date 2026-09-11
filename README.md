# whisper-turbo.c

## ASR + diarization cost comparison

USD, checked **September 10, 2026**. Native latency is the median of three
requests to an already-running service after a discarded warm-up request.
**Startup, model initialization and warm-up are excluded from latency and cost.**
The recording is 27.27 seconds (`std30.wav`); an audio-hour estimate extrapolates
many short requests, not a tested one-hour upload.

| Implementation / hosting | Result | Cost per 27.27 s of audio | Cost per audio hour |
| --- | --- | ---: | ---: |
| whisper-turbo.c, InstaCloud (4 vCPU) | Speaker-labeled text; **35.74 s** warm HTTP | **~$0.00126** | **~$0.166** |
| [OpenAI `gpt-4o-transcribe-diarize`](https://developers.openai.com/api/docs/pricing) | Speaker-labeled transcript | ~$0.00273 | ~$0.36 |
| [AssemblyAI Universal-2 + diarization](https://www.assemblyai.com/pricing) | Speaker-labeled transcript | ~$0.00129 | $0.17 |
| [AssemblyAI Universal-3.5 Pro + diarization](https://www.assemblyai.com/pricing) | Speaker-labeled transcript | ~$0.00174 | $0.23 |

The native estimate assumes four fully utilized CPUs and **1.1 GB RAM** during
the measured request, at [InstaCloud's published rates](https://instacloud.com/pricing).
This is a consistent resource scenario, **not a metered bill or a measured RAM cap**.
An always-on service also pays for resident memory between requests; idle time,
network, storage, account fees and credits are outside the active-request estimate.
Hosted API rows use published prices, not measured requests. No equal-accuracy or
hosted latency comparison is claimed. See [calculations and assumptions](docs/cost-comparison.md).

## InstaCloud CPU scaling

Warm HTTP transcription with diarization on Intel Xeon 6975P-C, INT8 weights
and encoder activations. Each result is the median of three requests for the same
27.27-second recording; startup and warm-up are excluded.

| vCPUs / threads | Request time | Speedup vs. 4 CPUs | Estimated cost per clip | Estimated cost per audio hour |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 35.74 s | 1.00× | ~$0.00126 | ~$0.166 |
| 8 | 19.24 s | 1.86× | ~$0.00127 | ~$0.168 |

Costs assume full utilization of the listed CPUs and 1.1 GB RAM during each
request, at the rates above; idle costs are excluded. Eight CPUs finish sooner
at nearly the same estimated cost per clip. These are separate runs on shared
CPUs, not a guarantee of scaling on every workload. See the
[4-CPU records](benchmarks/results/single-pass/runs-4cpu.json) and
[8-CPU records](benchmarks/results/single-pass/runs.json).

## Resident CPU performance

InstaCloud **Intel Xeon 6975P-C, 4 vCPU, four threads**, INT8 weights and encoder activations,
27.27-second recording. Each path keeps its model loaded, discards one warm-up,
and measures three sequential requests. No restart or cache eviction occurs between
requests. Startup and warm-up are not included.

| Path | Work | Median request time | Measured range |
| --- | --- | ---: | ---: |
| HTTP API | ASR + alignment + diarization + JSON response | **35.74 s** | 35.54–36.30 s |
| Direct CLI | ASR + alignment + diarization + JSON rendering | **35.27 s** | 35.24–35.31 s |

Both paths produce identical output and perform one ASR window and one diarization
pass for this recording. Timings are based on small samples on shared CPUs.
See [resident benchmark protocol and records](benchmarks/results/single-pass/README.md)
and [API validation and limits](docs/http-api.md#validation). Broad accuracy,
sustained-load behavior and AMD performance remain unverified.

## Overview

Run Whisper large-v3-turbo speech recognition on CPU with a native C11 runtime,
INT8 model weights, and AVX2/AVX-512 acceleration. Model conversion and inference
require no Python or C++ dependencies.

The application provides a command-line transcriber and a [C HTTP server](docs/http-api.md)
with `POST /v1/audio/transcriptions`, multipart uploads, and JSON/text responses.
The endpoint supports a documented subset of the OpenAI transcription interface,
including `gpt-4o-transcribe-diarize`-shaped speaker-segment responses. All accepted
model names select local C inference, not OpenAI-hosted weights.

The [experimental C-only Community-1 diarizer](docs/diarization.md) can run
standalone or in the shared CLI/HTTP transcription pipeline. Whisper transcribes
the recording once; Community-1 diarizes it once. Cross-attention alignment assigns
the existing text to speakers without retranscribing turns. Overlapping voices
are not separated.

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
up to 120 seconds in bounded windows. See [API options and limits](docs/http-api.md).

## Audio and output

The combined CLI uses exactly the same inference and response code as the HTTP
diarization endpoint, including automatic language detection and the 120-second
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
detection and recordings up to 120 seconds.

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

## License

MIT, with Apache-2.0 components in the optional diarizer. See [LICENSE](LICENSE)
and [diarization attribution](THIRD_PARTY_DIARIZATION.md), plus
[alignment attribution](THIRD_PARTY.md). Based on the Whisper Turbo implementation in
[llm-in-c](https://github.com/baryhuang/llm-in-c).

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
