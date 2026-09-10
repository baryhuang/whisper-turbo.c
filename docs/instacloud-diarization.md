# InstaCloud transcription and diarization benchmarks

Measured on September 10, 2026, on an Intel Xeon 6975P-C in InstaCloud's
`us-east` region. The allocation was eight shared vCPUs and 8192 MiB; each
inference command used eight threads. The guest exposed nine logical CPUs.

## Results

All four workloads used the same private 27.27-second mono PCM16, 16 kHz recording
(`std30.wav`). Each row contains three independent cold-model runs. Memory columns
show the highest observed peak, not the median. MB in the root README is decimal.

| Workload | Trial wall times, seconds | Median, seconds | Maximum process RSS, bytes | Maximum cgroup memory, bytes |
| --- | --- | ---: | ---: | ---: |
| whisper-turbo.c transcription | 17.989528, 13.908325, 13.541855 | 13.908325 | 939,753,472 | 955,105,280 |
| whisper.cpp Q8_0 transcription | 14.167229, 12.633593, 12.607642 | 12.633593 | 1,060,601,856 | 1,962,946,560 |
| Community-1 C diarization | 9.422000, 8.261567, 8.600754 | 8.600754 | 70,438,912 | 78,372,864 |
| Native transcription, then C diarization | 22.535885, 22.167647, 25.392566 | 22.535885 | 940,040,192 | 955,654,144 |

The combined row measures both executables sequentially under one observer. It
is not a sum of separately measured medians. Whisper exits before the diarizer
starts; their model-file caches remain accounted for. Process RSS is the largest
child-process high-water mark, while cgroup memory includes all descendants and
charged file cache. This is not a concurrently resident service measurement.

The diarizer returned one speaker for this recording, with identical JSON output
across its three standalone runs. Native transcripts were nonempty and identical
across their three runs. Repeatability is not an accuracy evaluation.

A separate 26-second two-voice A–B–A fixture completed in **7.381488 seconds**,
with **70,266,880 bytes** peak RSS and **78,168,064 bytes** cgroup memory. The C
identity check confirmed two speakers, the same label in both A sections, and a
different label for B. This is a single smoke test, not a diarization-error-rate
benchmark or an overlap-accuracy test.

Raw counters, run order, and speaker counts are in
[runs.json](../benchmarks/results/instacloud-diarization/runs.json).
Audio, transcripts, and detailed speaker timelines are not distributed.

## CPU paths and build

Native Whisper uses its existing AVX-512/VNNI path with packed attention, INT8
weights, and experimental INT8 encoder activations. Decoder activations remain
FP32. It uses the full 30-second encoder window and a 448-token decoder context.
The packed model is 847,660,800 bytes.

The diarizer uses **FP32 AVX2/FMA dot products and OpenMP convolutions**. It does
not currently use AVX-512, VNNI, or quantized neural-network weights. Its four
original Community-1 files total 32,820,977 bytes. Model revisions and checksums
are listed in [the diarizer documentation](diarization.md).

The Linux build uses GCC 12.2.0:

```sh
make diarization build/diarization-test build/observe OPENMP=-fopenmp \
  CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic -Werror'
```

The native diarizer links only the C runtime, libm, zlib, and libgomp. No Python,
PyTorch, ONNX Runtime, or C++ library is required. C checkpoint/kernel checks and
the A–B–A identity check pass on this Xeon. Local C checks also pass
AddressSanitizer and UndefinedBehaviorSanitizer on Apple Silicon.

See [CPU configuration](../benchmarks/results/instacloud-diarization/xeon-environment.txt)
and [Linux validation, source hashes, and binary hashes](../benchmarks/results/instacloud-diarization/validation.txt).
This verifies the current AVX2/FMA diarizer on Intel, not an AVX-512 diarizer or
AMD EPYC performance.

whisper.cpp is an external, CPU-only benchmark comparator, not a dependency of
the C application. It uses revision `c44b60b8053bbf2a5c1e014f11323fb3f2485177`,
its Q8_0 model (874,188,075 bytes), eight threads, English greedy decoding, and
its feature-compatible `zen4` backend. Its build and model identities are in
[the transcription benchmark report](instacloud-int8.md#reproduction).
It performs no Community-1 diarization in this comparison.

## Measurement method

The benchmark service is `kernel-bench` in project `whisper-bench`, existing cloud
branch `x86-q4`. Its image is pinned to:

```text
728301184821.dkr.ecr.us-east-1.amazonaws.com/insta-compute-builds-us-east-1@sha256:e06d66e890ea573cd0dd1543bfd8bbafcbc4d3dd609270fd47e9f478b2035c41
```

The diarizer and updated observer were compiled separately on that machine and
stored under `/data/community1-bench/src/build`; they are not in the pinned image.
The service was restarted and its health endpoint checked before **each** run.
No other remote execution overlapped inference.

The C observer evicts every relevant model file and checks residency with
`mincore` before starting the clock. All 13 successful observations began with
zero resident model pages and no configured swap. Prior cgroup lifetime peaks
were below 8 MB in every run, and each workload established a new peak. Both
prior and final counters are retained in the raw results.

Wall time includes process startup, model loading/validation, inference, and
output. It excludes model downloading, compilation, explicit cache eviction,
VM startup, health checks, and CLI/network round-trip latency. Shared-host timing
varies; the three trials are not a broad throughput study.

Cgroup peaks include reclaimable model-file cache and must not be interpreted
as minimum required RAM. The service allocation was not a hard 1.5-GB memory-cap
test. The observed sequential pipeline is below the project's decimal
1,500,000,000-byte target, but HTTP request handling, queues, concurrent resident
models, long recordings, and word-to-speaker alignment are not exercised.

## Run the same workloads

The following commands run inside the benchmark machine after supplying your
own WAV and the separately licensed model files. The original private audio is
not included; other recordings will produce different timings.

```sh
DIAR_MODELS=/data/community1-bench/models
DIAR_BIN=/data/community1-bench/src/build/diarize-community
OBSERVE=/data/community1-bench/src/build/observe
AUDIO=/path/to/mono-16khz.wav

# Transcription: native C.
"$OBSERVE" /data/turbo-q8.whtrbo \
  env OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE WHISPER_ACTIVATIONS=int8 \
  /app/whisper-turbo-x86 /data/turbo-q8.whtrbo "$AUDIO" 448 fixed30

# Transcription: external whisper.cpp baseline.
"$OBSERVE" /data/cpp-q8.bin /reference/build/bin/whisper-cli \
  -m /data/cpp-q8.bin -f "$AUDIO" -t 8 -l en -bo 1 -bs 1 -nt -nf -ng -sns

# Standalone C diarization.
"$OBSERVE" --models 4 \
  "$DIAR_MODELS/segmentation-pytorch_model.bin" \
  "$DIAR_MODELS/embedding-pytorch_model.bin" \
  "$DIAR_MODELS/plda-plda.npz" "$DIAR_MODELS/plda-xvec_transform.npz" \
  env OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$DIAR_BIN" "$DIAR_MODELS" "$AUDIO"

# Both CLIs sequentially, under one memory observer.
"$OBSERVE" --models 5 /data/turbo-q8.whtrbo \
  "$DIAR_MODELS/segmentation-pytorch_model.bin" \
  "$DIAR_MODELS/embedding-pytorch_model.bin" \
  "$DIAR_MODELS/plda-plda.npz" "$DIAR_MODELS/plda-xvec_transform.npz" \
  env OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE WHISPER_ACTIVATIONS=int8 \
  sh -c '/app/whisper-turbo-x86 /data/turbo-q8.whtrbo "$1" 448 fixed30 && "$2" "$3" "$1"' \
  pipeline "$AUDIO" "$DIAR_BIN" "$DIAR_MODELS"
```

For independent memory peaks, restart the isolated service before **each command**
using `insta --agent compute restart kernel-bench --branch x86-q4`, verify health,
then run exactly one workload with `insta --agent compute exec ... --timeout 180`.
Do not run all four commands in one unchanged VM lifetime and treat the resulting
counters as independent peaks.
