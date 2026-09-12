# CPU inference optimizations

Measured September 11, 2026 on one InstaCloud Intel Xeon 6975P-C, eight shared
vCPUs, eight inference threads and one resident model. Baseline is `f65795f`;
candidate is `59051c7` plus the CPU changes described below.

Both builds set `WHISPER_ACTIVATIONS=int8` with INT8 weights. The candidate also
sets `WHISPER_DECODER_ACTIVATIONS=int8`, `WHISPER_DIAR_SIMD=avx512` and
`WHISPER_DIAR_BATCH_INPUT=1`. **The baseline is already using opt-in INT8 encoder
activations; this table does not measure the default FP32-activation server.**
See the [startup command](../README.md#cpu-options) for these settings.

| Workload | Baseline | Candidate | Less processing time | Output check |
| --- | ---: | ---: | ---: | --- |
| LibriSpeech test-clean rows 0–9, 91.525 seconds audio | 107.110 s | 92.605 s | 13.5% | Identical transcripts; 3 errors / 251 reference words (1.1952% WER) |
| Synthetic two-speaker A–B–A, 26 seconds, ASR + diarization + alignment | 18.799 s | 14.876 s | 20.9% | Byte-identical JSON, including text, timestamps and A–B–A labels |

ASR timing is the sum of ten sequential HTTP requests after one discarded
warm-up. Combined timing is the mean of two requests after one discarded
warm-up per engine. These small, English-language tests are not a general
accuracy, DER, or production-cost claim. Both engines ran sequentially on the
same machine; shared-host performance can vary.

At a common scenario of eight fully utilized vCPUs and 1.1 GB RAM, estimated
active-processing cost per audio hour falls **$0.278 → $0.240 for ASR** and
**$0.172 → $0.136 for combined processing**. Each uses its own measured audio
duration; these are not metered production savings. See
[rates, arithmetic and exclusions](cost-comparison.md).

The production integration also completed four read-only real hub/watch tests,
totaling **910 seconds of audio**, with no inference, alignment, diarization or
privacy errors. Full-service peak was **1,441,619,968 bytes**, including the
adapter, resident C model, request handling and charged file cache; no OOM or
swap was observed. The mixed-language five-minute watch preserved both
languages, two speakers and the final speech at 298.98 seconds. This is a
regression check, not human-labeled recognition or diarization accuracy.

The integration deployed image
`public.ecr.aws/j2i4i5w2/caremojo-transcribe@sha256:afa45f25140ae1f65ea135c4859e7140411fd761632d50fb96ed2c8375e24f98`.
Its native binary SHA-256 is
`b9be9869815d43f8d39bdbd4030044b65a57c6aba3ac283a744cbb252c1d1631`.
Raw private recordings and transcripts are excluded from this repository.

## Available options

| Setting | Effect | Default |
| --- | --- | --- |
| `WHISPER_ACTIVATIONS=int8` | Q8 weights and INT8 encoder activations | Off (FP32 activations) |
| `WHISPER_DECODER_ACTIVATIONS=int8` | INT8 activation matrix/vector operations in decoder projections and vocabulary head | Off (FP32 activations) |
| `WHISPER_DIAR_SIMD=avx512` | FP32 AVX-512/FMA diarization dot products on capable CPUs; AVX2/scalar fallback otherwise | Existing AVX2/scalar path |
| `WHISPER_DIAR_BATCH_INPUT=1` | Parallel input projections across LSTM time steps; recurrent updates remain sequential | Off |

Set these before starting the service. Diarization SIMD selection is cached
per thread at first use. INT8 encoder and decoder activations are independently
opt-in pending wider accuracy validation; they do not quantize attention,
normalization, or diarization weights to INT8.

The W8A8 kernel uses a bounded 16-row workspace of **84,480 bytes** instead of
allocating activation buffers per layer. Maximum input width is 5,120. Group
quantization, bias handling, accumulation order and invalid-input rejection
are tested against an independent reference, including actual AVX2/VNNI runs.
LSTM input batching uses approximately 1.2 MB of temporary workspace for a
ten-second segmentation window, freed after each direction. No second resident
ASR model or concurrent transcription is introduced.

The observed service-accounted lifetime memory peak across the combined
benchmark runs was **1,120,550,912 bytes**, including charged file cache. That
counter was not independently reset between engines. It is a bound for these
workloads, not a full-production memory acceptance result or proof of a hard
1.5-GB allocation. The platform reported a 2-GiB allocation; the guest exposed
more physical RAM and an unrestricted root cgroup.

## Sources and reproduction

The decoder work follows the INT8 activation/GEMM pattern used by
[CTranslate2](https://github.com/OpenNMT/CTranslate2/blob/v4.8.1/src/layers/common.cc),
the inference engine behind
[WhisperX](https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/asr.py).
This is not WhisperX's complete batched/VAD implementation: existing ASR
windowing and quality retries are preserved. LSTM batching and SIMD dispatch
are native implementation changes, not code copied from WhisperX.

Use [the C WER tools](../benchmarks/wer/README.md) and
[`run-case.sh`](../benchmarks/cpu/run-case.sh) for sequential ASR tests.
The existing [`diarization-fixture`](../tools/diarization_fixture.c) tool creates the synthetic fixture from
the first eight seconds of LibriSpeech `6930-75918-0001` and
`8455-210777-0030`, with one-second gaps; the first voice is repeated.
LibriSpeech is CC BY 4.0, by Vassil Panayotov, Guoguo Chen, Daniel Povey and
Sanjeev Khudanpur, using LibriVox recordings. Dataset revision:
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`.

[`fetch-diarization.sh`](../benchmarks/cpu/fetch-diarization.sh) downloads
checksum-pinned, separately licensed Community-1 checkpoints at runtime.
[`run-diarized.sh`](../benchmarks/cpu/run-diarized.sh) performs the combined
HTTP test. Keep the isolated instance awake during tests; an interrupted
exit-137 run was excluded, not counted as successful. No private recordings,
credentials, or gated checkpoints are included in benchmark images.
