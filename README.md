# whisper-turbo.c

Whisper large-v3-turbo extracted from
[llm-in-c](https://github.com/baryhuang/llm-in-c) at commit
`da2d24deb33c6663617d179522bfa1aebb95b2cf`.

The goal is a native C Whisper large-v3-turbo API service matching the OpenAI
transcription interface, using **less than 1.5 GB of total service memory**
on two x86-64 CPU targets:

| Target | Processor | Kernel direction |
| --- | --- | --- |
| `amd-epyc-avx2` | AMD EPYC, family 25 model 1 | AVX2, FMA, F16C |
| `intel-xeon-avx512` | Intel Xeon 6975P-C | AVX-512, with optional VNNI, BF16 and FP16 paths |

Both targets use an initial eight-thread inference budget. See the
[service target specification](docs/service-targets.md) for the memory accounting,
implementation plan, and acceptance criteria.

The implementation includes the generic C11 runtime, a streaming C INT8 model
importer, runtime-dispatched AVX2/AVX-512 kernels, SIMD attention, and experimental
INT8-activation/VNNI encoder kernels. The HTTP API is not implemented yet.
Inference benchmarks are not a certification of the future API's memory budget.

## Build and run

```sh
make
./build/whisper-turbo-transcribe /path/to/model.whtrbo speech.wav 96 compact
```

Requires a C11 compiler, Make, POSIX APIs, and libm. OpenMP is optional:
`make OPENMP=-fopenmp` with a supporting compiler. No Python or C++ is required
to build or run these binaries.

The inherited CLI accepts mono 16-bit PCM WAV at 16 kHz, forces English and
greedy decoding, and reads at most the first 30 seconds. Its output includes
benchmark logs. It is not yet suitable as an API transcription implementation.

Weights are not included. Import the published original GGML F16/F32 Turbo
checkpoint using C (prequantized GGML input is not accepted):

```sh
make x86-tools OPENMP=-fopenmp
./build/import-ggml ggml-large-v3-turbo.bin turbo-q8.whtrbo
OMP_NUM_THREADS=8 ./build/whisper-turbo-x86 turbo-q8.whtrbo speech.wav 96 fixed30
```

The original checkpoint's SHA-256 must be
`1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`.
The validated imported image is 847,660,800 bytes. The importer creates its
output exclusively: choose a new destination when retrying a failed conversion.

`WHISPER_SIMD=scalar|avx2|avx512` selects a supported path, with automatic
feature-checked dispatch by default. `WHISPER_ACTIVATIONS=int8` enables the
experimental W8A8 encoder; decoder activations remain FP32. Leave this unset
for the FP32-activation reference path. W8A8 passes the current kernel and JFK
checks, but needs a wider accuracy corpus before becoming a production default.

## Layout

- `src/generic/`: encoder, cached decoder, log-Mel frontend, image loader, quantization definitions.
- `src/whisper_turbo_transcribe.c`: inherited transcription CLI.
- `src/x86/`: runtime-dispatched C intrinsics and INT8 kernels.
- `tools/import_ggml.c`: streaming C checkpoint conversion.
- `benchmarks/`: C parity, performance and Linux memory-observation tools.
- `benchmarks/results/instacloud-int8/`: raw cloud experiment outputs.
- `docs/pins.json`: pinned checkpoint identities and architecture.
- `docs/upstream/`: historical Turbo results for Jetson Orin and RK3588.
  These are reference records from the original repository, not supported
  backends here. Their paths and reproduction commands refer to that repository.

A113X code and results are excluded. The remaining implementation is portable
CPU C; the historical C++/CUDA/RKNPU backends and all Python tooling are excluded.

The current [InstaCloud CPU inspection](docs/instacloud-cpu.md) identifies
x86-64 AMD EPYC and Intel Xeon workers configured for 8 shared vCPUs and 8 GiB.
AVX2 is common to the observed workers; the Intel workers also expose AVX-512.

## Development requirements

Implementation, model preparation, serving, and tests must be in C. Do not add
Python scripts, Python build dependencies, Python inference services, or C++
runtime backends. Makefiles, deployment configuration, and documentation are
allowed.

The root Dockerfile is a **benchmark-only** image: it additionally builds a
pinned external whisper.cpp comparison executable using that project's C++
toolchain. That executable is not linked into, invoked by, or required to build
the native C application. Neither benchmark path executes Python.

The API target is `POST /v1/audio/transcriptions`, with the multipart request
and response contract documented in the
[OpenAI transcription reference](https://developers.openai.com/api/reference/resources/audio/transcriptions/methods/create).
Track parameter support, audio decoding, language detection, prompt tokenization,
sampling, long audio, and timestamp alignment explicitly. Do not claim full
compatibility until tested, fabricate timestamps or probabilities, or silently
ignore unsupported requests. A local Turbo model cannot reproduce proprietary
OpenAI models merely by accepting their names.

## License

MIT. See [LICENSE](LICENSE). Upstream Git history is preserved; unrelated files
and excluded languages remain in historical commits, but not the current tree.
