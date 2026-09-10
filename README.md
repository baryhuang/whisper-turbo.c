# whisper-turbo.c

Run Whisper large-v3-turbo speech recognition on CPU with a native C11 runtime,
INT8 model weights, and AVX2/AVX-512 acceleration. Model conversion and inference
require no Python or C++ dependencies.

The application provides a command-line transcriber. An HTTP server and
OpenAI-compatible transcription endpoint are not available.

## Quick start

Requires a C11 compiler, Make, POSIX APIs, and libm. For multithreaded x86 inference,
use a compiler with OpenMP support:

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

## Audio and output

Input must be mono 16-bit PCM WAV at 16 kHz. The transcriber processes only the
first 30 seconds, uses English greedy decoding, and prints text on a `TRANSCRIPT:`
line alongside timing logs. Language detection, timestamps, and long-audio
processing are not supported.

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

## Performance

On an Intel Xeon 6975P-C with eight threads, the 30-second repeated-speech
benchmark produced these cold-run results:

| Implementation | Model size | Peak process RSS | Peak cgroup memory | Wall time |
| --- | ---: | ---: | ---: | ---: |
| whisper-turbo.c, INT8 encoder activations | 847.7 MB | 939.1 MB | 955.3 MB | 18.94 s |
| whisper.cpp Q8_0 | 874.2 MB | 1,060.8 MB | 1,963.7 MB | 17.64 s |

Units are decimal. The native inference peak was below 1.5 GB. Cgroup memory
includes reclaimable model-file cache; it is not the same as process RSS or a
minimum memory requirement. Wall time includes cold model loading. These
single-run measurements do not establish a general speed advantage or an HTTP
service memory guarantee.

AVX2 and AVX-512 paths were tested on Xeon hardware; AMD EPYC performance is not
verified. See [benchmark results and methodology](docs/instacloud-int8.md).

## Tests

```sh
make check
```

Runs deterministic C checks for quantized kernels and attention parity.

The root Dockerfile builds a benchmark image containing an external, pinned
whisper.cpp comparison executable. Its C++ toolchain is not required by the
native application.

## License

MIT. See [LICENSE](LICENSE). Based on the Whisper Turbo implementation in
[llm-in-c](https://github.com/baryhuang/llm-in-c).
