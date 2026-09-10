# Whisper API service targets

## Objective

Create a C implementation of the OpenAI `POST /v1/audio/transcriptions`
interface backed by Whisper large-v3-turbo, with peak total service memory
**strictly below 1,500,000,000 bytes** on both supported CPU targets.
The threshold uses decimal GB, approximately 1430.5 MiB, not 1.5 GiB.

This document sets development and release requirements. It does not claim an
implemented HTTP server, completed CPU optimizations, or measured memory compliance.

## CPU targets

| Target | Observed hardware | Required optimized baseline | Optional acceleration |
| --- | --- | --- | --- |
| `amd-epyc-avx2` | AMD EPYC, family 25 model 1 | x86-64, AVX2, FMA, F16C | None assumed |
| `intel-xeon-avx512` | Intel Xeon 6975P-C | x86-64, AVX2/FMA/F16C; AVX-512 F/DQ/BW/VL | VNNI, BF16, FP16 when individually detected and validated |

Both were observed on InstaCloud services configured for 8 shared vCPUs and
8192 MiB. Those existing allocations are not the memory target for this service.
Start with eight inference threads and allow a lower explicit thread count.
The Intel guests exposed nine CPUs; do not use that to silently raise the
eight-thread budget. Record the actual host CPU flags during each benchmark.

Retain generic C kernels for correctness comparisons and unsupported machines.
Dispatch specialized kernels only after checking CPU features and OS vector-state
support. Keep specialized instructions out of baseline startup code. Build
target-specific objects so an AVX-512 build machine cannot accidentally make
the entire executable incompatible with AMD. Hardware observations are in
[instacloud-cpu.md](instacloud-cpu.md); machine-readable target definitions
are in [`targets/`](../targets/).

## Memory design

Use one resident model and one active transcription per service instance
initially. Keep the pending queue bounded, with bounded request metadata.
Do not spawn a new model-loading inference process for each HTTP request.
Reject overload explicitly before reserving another inference workspace.

Start by evaluating packed Q4 weights; compare Q5 if quality requires it.
The existing C core already represents Q4/Q5/Q8 matrices. Quantization is a
quality decision and must pass a fixed transcription regression corpus on both
targets. Keep quantized weights packed and dequantize small tiles; never expand
the complete checkpoint into floating-point weights in the serving process.

The following is a planning budget, not an allocation measurement:

| Component | Budget, decimal MB |
| --- | ---: |
| Resident model, tokenizer and model metadata | 600 |
| Encoder/decoder workspaces and KV caches | 350 |
| Audio decoding, resampling and window buffers | 100 |
| HTTP uploads, bounded queue and response buffers | 100 |
| Executable, libraries, thread stacks and allocator overhead | 150 |
| Planned working total | 1300 |

Leave the remaining 200 MB as headroom below the strict 1500 MB threshold.
If measurements exceed a component budget, revise allocation strategy while
preserving the total limit. Include SIMD packing buffers, per-thread scratch
space, and decoder timestamp-alignment buffers in these budgets.

Bound audio decoding and process long recordings in model-sized windows without
truncation. Bound in-memory multipart buffers and avoid retaining both the whole
compressed upload and the whole decoded waveform. Disk-backed uploads still
consume page cache; account for that usage. Do not use swap to conceal an
oversized working set. Allocation failures must produce an explicit API error
without leaving the resident service unusable.

## API scope and implementation sequence

1. Add a C model importer and validate packed weights and tokenizer metadata.
2. Extract a reusable resident inference API from the CLI and add bounded audio
   processing, language detection, prompt tokenization, sampling, and long-audio
   decoding.
3. Add the C HTTP transcription endpoint, multipart parsing, request validation,
   OpenAI response/error structures, and the required audio format decoders.
4. Implement and verify text/JSON, verbose JSON, SRT and VTT responses and real
   segment/word alignment where required by the Whisper API contract.
5. Add and validate AVX2 and AVX-512 kernels against generic C, then measure
   service memory, accuracy, and throughput on both CPU targets.

The compatibility target is the Whisper transcription contract. Explicitly
document the local Turbo model identity and any model alias. Do not claim
proprietary OpenAI model equivalence or fabricate unsupported output fields.
Full interface compatibility is an acceptance requirement; unsupported features
must remain visible gaps until implemented, not silently ignored parameters.
Implementation, model preparation, serving, and test programs must be C, without
Python or C++ build/runtime dependencies. This specification sets requirements;
it does not authorize deployment or resizing existing InstaCloud services.

## Acceptance evidence

Measure the complete service in a dedicated Linux cgroup with no swap. Record
cgroup peak memory, including anonymous allocations and charged file cache,
alongside process peak RSS and all child processes. If kernel peak counters are
unavailable, sampled RSS alone is insufficient to claim the strict ceiling.
Use the same memory-accounting method on both CPU targets.

Cover cold model load, warm repeated requests, short and full-window audio,
long recordings, supported compressed formats, maximum-size uploads, multilingual
input, prompts, timestamp modes, and concurrent arrivals at the queue limit.
Assert bounded resource use for malformed requests, cancellations and rejected
overload. Run sustained requests to check for memory growth. Do not declare
success based only on a short clip, packed file size, or an idle RSS reading.

For each target, retain the exact build, CPU flags, thread count, weight hash,
quantization, corpus and request options, correctness results, peak memory in
bytes, and end-to-end latency/throughput. Pass requires all supported workloads
to remain strictly below 1,500,000,000 bytes, with no OOM, swap, silent truncation,
or unexplained accuracy regression. Performance comparisons apply only after
the API, quality, and memory gates pass. No latency target is set yet.
