# InstaCloud INT8 inference experiments — 2026-09-10

## Scope and outcome

This is a C inference benchmark milestone, **not an implemented transcription
HTTP API or a certification of the future API's memory limit**. The C importer,
runtime-dispatched x86 kernels and benchmark programs contain no Python. The
external whisper.cpp executable is used only as a comparison; it is not a native
runtime dependency.

The native packed model is **847,660,800 bytes**, versus **874,188,075 bytes** for
whisper.cpp Q8_0 prepared from the same original checkpoint (3.03% smaller).
Fresh-run inference memory was below the strict decimal 1,500,000,000-byte target.
The raw records include failures and inconclusive memory runs as well as passes.

## Fresh-run measurements

Final packed-attention W8A8 runs allocate the full 448-token decoder context.
JFK is 11 seconds of actual audio padded to the full 30-second encoder window.
The stress fixture contains 30 seconds of repeated speech, not merely padding.
All listed runs exited successfully, started with zero resident model pages and
reported no configured swap. Times include process startup and cold model load.

| Instance / workload | Implementation | Cgroup peak, bytes | Process peak RSS, bytes | Wall time, seconds |
| --- | --- | ---: | ---: | ---: |
| Virginia / JFK | Native, default FP32 activations, before K/V packing | 936,349,696 | 920,653,824 | 28.43 |
| Virginia / 30-second stress | Native W8A8, packed attention | 955,305,984 | 939,143,168 | 18.94 |
| Virginia / 30-second stress | whisper.cpp Q8_0 | 1,963,679,744 | 1,060,761,600 | 17.64 |
| Frankfurt / JFK, 4 threads | Native W8A8, AVX2-only, packed attention | 951,230,464 | 936,353,792 | 33.64 |
| Frankfurt / JFK, 4 threads | whisper.cpp Q8_0 | 1,959,821,312 | 1,058,856,960 | 29.48 |
| Singapore / JFK, 8 threads | Native W8A8, VNNI, packed attention | 951,492,608 | 936,058,880 | 16.65 |
| Singapore / JFK, 8 threads | whisper.cpp Q8_0 | 1,960,849,408 | 1,058,320,384 | 14.22 |

For these matched workloads, native cgroup peaks were approximately **51% lower**
and process peak RSS approximately **11.5% lower**. The packed model file is also
smaller. This is not a blanket speed victory: the single-run timing comparisons
above favor whisper.cpp. Shared-host variability is substantial.
The cgroup difference includes whisper.cpp's charged input-file cache alongside
its loaded tensors. That cache is reclaimable; 1.96 GB is the observed cold-run
peak, not a claim that whisper.cpp has an irreducible 1.96-GB working set.

Separately, six interleaved native JFK runs in one unchanged instance used
packing settings `0,1,1,0,0,1`. Median wall time was **13.4457 seconds unpacked**
versus **10.2038 seconds packed** (24.1% reduction), with identical transcripts.
The previously recorded 955,305,984-byte lifetime peak did not increase in any
of those runs; that is an upper bound, not six independent reset peaks.

Raw final records: [Virginia native](../benchmarks/results/instacloud-int8/xeon-packed-repeat30.txt),
[Virginia baseline](../benchmarks/results/instacloud-int8/xeon-cpp-repeat30.txt),
[Frankfurt native](../benchmarks/results/instacloud-int8/eu-packed-final.txt),
[Frankfurt baseline](../benchmarks/results/instacloud-int8/eu-cpp-1280m.txt),
[Singapore native](../benchmarks/results/instacloud-int8/ap-packed-final.txt),
[Singapore baseline](../benchmarks/results/instacloud-int8/ap-cpp-fixed30.txt),
[packing interleave](../benchmarks/results/instacloud-int8/xeon-packing-interleaved.txt).

## Hardware survey

Project: `whisper-bench` (`af78250c-3790-44b0-8441-feaf4cefde30`).
Isolated cloud branch: `x86-q4` (the name predates the INT8 requirement).
Git branch: `bench/x86-int8`.

| New service | Advertised region | CPU observed | Configuration used |
| --- | --- | --- | --- |
| `kernel-bench` | Virginia / `us-east` | Intel Xeon 6975P-C | 8 shared vCPU, 8192 MiB |
| `cpu-survey-eu` | Frankfurt / `eu-central` | Intel Xeon 6975P-C | Initially 8/8192; then 4 vCPU, 1280 MiB |
| `cpu-survey-ap` | Singapore / `ap-southeast` | Intel Xeon 6975P-C | 8 shared vCPU, 8192 MiB |

The two additional services were created after the first successful native
sub-1.5-GB comparison. All three exposed AVX2/FMA and AVX-512/VNNI/BF16/FP16.
The AVX2-only path was also forced explicitly, including on the smaller Frankfurt
allocation. **No AMD CPU appeared in these newly provisioned instances.** This
validates AVX2 functionality on Intel, not AMD performance. The earlier EPYC
observation remains a deployment target; it is not substituted with invented AMD
results. Three samples do not establish exhaustive coverage of all cloud hosts.

Raw CPU and guest information: [Virginia](../benchmarks/results/instacloud-int8/xeon-environment.txt),
[Frankfurt](../benchmarks/results/instacloud-int8/eu-environment.txt),
[Singapore](../benchmarks/results/instacloud-int8/ap-environment.txt).

## Implemented optimizations

- Q8 weights: 128 signed bytes plus one BF16 scale per group; weights stay packed.
- FP32-activation path: AVX2/FMA and AVX-512 widening/FMA group dots, with CPU/OS
  feature checks and scalar fallback. No global `-march=native` assumption.
- Experimental W8A8 encoder: activations quantized per 128 values, AVX2 widened
  integer multiply-add or AVX-512 VNNI dot products. Unsigned VNNI activation
  offsets are corrected exactly; no saturating byte-dot shortcut is used.
- Attention: SIMD double-precision dot/context accumulation and head-major K/V
  packing. Scratch is bounded to 15,360,000 bytes for a full window; allocation
  failure falls back to the original layout. `WHISPER_ATTENTION_PACK=0` disables
  packing for comparisons.
- The decoder retains FP32 activations. `WHISPER_ACTIVATIONS=int8` is opt-in;
  the default remains the more conservative FP32-activation path.

BF16/FP16 instructions are not enabled merely because their flags exist:
changing attention precision would require additional accuracy evidence. The
current INT8 arithmetic uses VNNI where useful and keeps validated accumulation
precision elsewhere.

## Correctness and kernel checks

`make check` runs C-only deterministic tests. Local Clang builds pass
`-Wall -Wextra -Wpedantic -Werror` and AddressSanitizer/UndefinedBehaviorSanitizer;
cloud GCC builds pass the warning-as-error configuration too.

- W8A8 integer dots: 10,000 vectors per run, including extreme signed values;
  scalar, AVX2 and VNNI results are exactly equal. Quantized GEMM outputs match
  the scalar integer reference byte-for-byte.
- Activation quantization on the synthetic matrix tests: normalized RMS error
  about 0.0039 versus FP32 activations. This is **not a speech accuracy metric**.
- Attention: 1, 7, 64, 550 and 1500 frames; the tested scalar/AVX2/AVX-512 outputs
  match the inherited double-accumulation reference exactly, including packing.
- Full model: the JFK clip transcribes correctly with the default path, AVX2
  W8A8 and VNNI W8A8. A C-generated 30-second repetition exercises full-window
  audio and a longer output with a 448-token decoder allocation. It is synthetic
  stress audio, not an independent accuracy corpus.

For the 1500×1280 input / 5120 output matrix at eight threads, the clean
FP32-activation microbenchmark medians were 704.5 ms (inherited generic), 190.1 ms
(AVX2), and 187.2 ms (AVX-512). VNNI W8A8 trials were about 43–44 ms in its
separate experiment; these change activation precision and are not an exact
arithmetic replacement. Packing reduced the full-window attention test from
216.1 ms to 93.8 ms on the same Singapore instance. These are kernel timings,
not promises about end-to-end speed.

Evidence: [clean Q8 kernels](../benchmarks/results/instacloud-int8/xeon-q8-kernels-clean.jsonl),
[W8A8 kernels](../benchmarks/results/instacloud-int8/xeon-w8a8-kernels.jsonl),
[attention packing/parity](../benchmarks/results/instacloud-int8/ap-packed-attention-validation.txt),
[local sanitizers](../benchmarks/results/instacloud-int8/local-sanitizer-validation.txt).
The packing/parity log tests packing off first, then on, at 64/550/1500 frames.

## Memory methodology and caveats

`benchmarks/cloud/observe.c` reads `memory.peak` in the execution's cgroup,
including charged file cache and descendant processes, and reports process peak
RSS separately. It refuses configured swap. Before starting inference it asks
the kernel to evict the specified model file and verifies residency with
`mincore`; accepted measurements here started with zero resident model pages.

The counter is a **lifetime peak, not a per-run reset**. The guest rejects
nested-controller setup (`EBUSY`) and memory-limit writes (`EPERM`). Fresh memory
comparisons therefore restart/redeploy the isolated service before each side,
then run the observer inside that same execution. A separate later `exec cat`
is not a substitute for measuring inside the benchmark. The observer prints the
prior peak and initial current usage so contaminated runs remain visible.

The [kernel documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
describes cgroup peak accounting and reset semantics. `measure.c` is an optional
runner for separately delegated Linux cgroups; its mutation modes are not usable
on these InstaCloud guests. The read-only observer is the tool used for results.

The API does not exist yet. These measurements include the benchmark execution
and its charged cache, but do not exercise HTTP uploads, request queues, codec
processes, resident-service reuse, cancellations or concurrent requests. Do not
present them as complete-service API acceptance.

InstaCloud rejected 8 vCPU / 1280 MiB because its legal minimum was 2048 MiB.
It accepted 4 vCPU / 1280 MiB. That is a **configuration observation, not proof
of a hard cgroup ceiling**: the guest reported `memory.max=max`, and the
comparison run's accounted peak exceeded that configured size without failing.
Native compliance is based on the measured peak, not that platform setting.

One overlapping execution returned 137 with no benchmark output; its cause was
not established. The subsequent checked run reported zero OOM events. Keep the
empty `xeon-native-w8a8-clean.txt` and the cache-contaminated retry as diagnostics,
not acceptance evidence. Later trials are serialized per service, with temporary
always-on enabled during tests. Model preparation, compiler runs, hashing both
model files and an earlier comparison can all inflate a lifetime peak.

## Reproduction

Runtime source with packed attention: `55c1ea4`.
Benchmark image source: `1ec5c41a7786129296f4673811e228f6ec93d821`.
Image:
`728301184821.dkr.ecr.us-east-1.amazonaws.com/insta-compute-builds-us-east-1@sha256:e06d66e890ea573cd0dd1543bfd8bbafcbc4d3dd609270fd47e9f478b2035c41`.
whisper.cpp: `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, Release,
`GGML_NATIVE=OFF`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_BACKEND_DL=ON`, CPU-only.
It selected its `zen4` feature-compatible CPU backend on these Intel guests;
no scalar-only baseline was forced. Flash attention remained enabled by default.

Native build:

```sh
make -j8 all x86-tools build/observe build/repeat-wav OPENMP=-fopenmp
make check OPENMP=-fopenmp
./build/import-ggml ggml-large-v3-turbo.bin turbo-q8.whtrbo
```

Original model: [published GGML Turbo checkpoint](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo.bin).
SHA-256: `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`.
Native Q8: `bab4e97984f15276de3b7dbdce70624b0600d102619ce561d0a35cf9f2bba3b3`.
whisper.cpp Q8_0: `317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1`.
JFK WAV: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
Repeated-30s WAV: `712169d8a78022fdda7d946aa478e3bbeea7f7dc75608d7327b4cc2a65a266cb`.

After model preparation, restart the service, verify its health URL, then run
one benchmark execution at a time. Example native run:

```sh
insta --agent compute exec kernel-bench --branch x86-q4 --timeout 180 -- \
  /app/observe /data/turbo-q8.whtrbo env OMP_NUM_THREADS=8 \
  WHISPER_ACTIVATIONS=int8 /app/whisper-turbo-x86 \
  /data/turbo-q8.whtrbo /app/jfk.wav 448 fixed30
```

Use `WHISPER_SIMD=avx2` to restrict instructions and `OMP_NUM_THREADS=4` for the
Frankfurt configuration. Omit `WHISPER_ACTIVATIONS` for the default path.
For the baseline, restart again and run:

```sh
insta --agent compute exec kernel-bench --branch x86-q4 --timeout 180 -- \
  /app/observe /data/cpp-q8.bin /reference/build/bin/whisper-cli \
  -m /data/cpp-q8.bin -f /app/jfk.wav -t 8 -l en \
  -bo 1 -bs 1 -nt -nf -ng -sns
```

Both sides use the same audio, Turbo checkpoint family, INT8 weights, English,
greedy decoding and full encoder window. Their quantization group sizes differ;
identical precision labels do not imply identical tensor values. Latencies on
shared vCPUs vary; repeated measurements are needed for performance claims.

## Remaining release gates

Actual AMD-host benchmarking; a wider independent multilingual accuracy corpus;
resident C inference API and bounded long-audio processing; HTTP multipart and
OpenAI-compatible response/error behavior; real timestamps; complete-service
memory/load/overload tests. W8A8 stays experimental until its accuracy gate passes.
