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

The current implementation contains the inherited generic C11 runtime, CLI,
and encoder benchmark. The HTTP API and optimized x86 kernels are not yet
implemented, and the service memory target has not yet been validated.

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

Supply an existing packed `.whtrbo` model separately. Weights are not included.
The original Python model converter has been excluded; a C model importer is
required before this repository can prepare checkpoints on its own.

## Layout

- `src/generic/`: encoder, cached decoder, log-Mel frontend, image loader, quantization definitions.
- `src/whisper_turbo_transcribe.c`: inherited transcription CLI.
- `benchmarks/`: C encoder benchmark.
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
