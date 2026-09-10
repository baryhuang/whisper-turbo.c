# whisper-turbo.c

## CPU performance

Measured on InstaCloud: **Intel Xeon 6975P-C, eight threads**, using the same
27.27-second recording (`std30.wav`). Times are medians of three cold-model runs;
memory figures are the highest peaks observed across those runs.

| Implementation | Task | Wall time | Peak process RSS | Peak cgroup memory |
| --- | --- | ---: | ---: | ---: |
| whisper-turbo.c | Transcription | 13.91 s | 939.8 MB | 955.1 MB |
| whisper.cpp Q8_0 | Transcription | 12.63 s | 1,060.6 MB | 1,962.9 MB |
| Community-1 in C | Speaker diarization | 8.60 s | 70.4 MB | 78.4 MB |
| whisper-turbo.c + Community-1 in C | Transcription, then diarization | 22.54 s | 940.0 MB | 955.7 MB |

Units are decimal. Native transcription uses opt-in INT8 encoder activations and
AVX-512/VNNI; the diarizer uses FP32, AVX2/FMA, and OpenMP. The combined row is a
measured sequential execution of two CLIs, not an HTTP service or a
speaker-attributed transcript. whisper.cpp is a transcription-only baseline.

Wall time includes process startup and cold model loading, but excludes VM
startup. Cgroup memory includes reclaimable file cache; it is not a minimum RAM
requirement. These short-clip results do not establish general accuracy,
long-audio performance, or an HTTP-service memory guarantee. AMD EPYC performance
is not verified. See [measurements and reproduction](docs/instacloud-diarization.md)
and [additional transcription benchmarks](docs/instacloud-int8.md).

The **resident HTTP server** was also tested on the same Xeon with eight threads:

| HTTP workload | Request time |
| --- | ---: |
| 11-second JFK, default FP32 activations, cold weights | 73.23 s |
| 11-second JFK, INT8 activations, warm weights | 10.76 s |
| 120-second repeated speech, INT8 activations | 44.76 s |
| 24,999,000-byte WAV upload, 11 seconds of speech, INT8 activations | 10.73 s |

These are individual observations, not medians or matched precision comparisons.
The highest completed HTTP-test peak was **1,030.0 MB cgroup memory** and
**1,012.1 MB process RSS**, including the near-limit upload test. Repeated requests,
automatic language detection, and speech after 30 seconds passed. See
[HTTP validation and limits](docs/http-api.md#validation); these results do not
certify full API compatibility or all production workloads.

## Overview

Run Whisper large-v3-turbo speech recognition on CPU with a native C11 runtime,
INT8 model weights, and AVX2/AVX-512 acceleration. Model conversion and inference
require no Python or C++ dependencies.

The application provides a command-line transcriber and a [C HTTP server](docs/http-api.md)
with `POST /v1/audio/transcriptions`, multipart uploads, and JSON/text responses.
The endpoint supports a documented subset of the OpenAI Whisper interface;
compressed audio, prompts, and timestamp/diarized response formats are not yet supported.

An [experimental C-only Community-1 diarizer](docs/diarization.md) also provides
timestamped speaker labels. It runs separately from transcription; automatic
speaker assignment to transcript words is not supported.

## HTTP transcription

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=whisper-1 -F language=en
```

Returns `{"text":"..."}`. Add `-F response_format=text` for plain text. `whisper-1`
is an alias for the local Turbo model, not OpenAI's hosted model. The server keeps
one model loaded, accepts one transcription at a time, and processes WAV recordings
up to 120 seconds in bounded windows. See [API options and limits](docs/http-api.md).

## Audio and output

Input must be mono 16-bit PCM WAV at 16 kHz. The command-line transcriber processes only the
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
and [diarization attribution](THIRD_PARTY_DIARIZATION.md). Based on the Whisper Turbo implementation in
[llm-in-c](https://github.com/baryhuang/llm-in-c).

## Setup and compilation

Requires a C11 compiler, Make, POSIX APIs, and libm. For multithreaded x86 inference,
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
