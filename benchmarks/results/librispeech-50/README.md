# LibriSpeech test-clean: 50-utterance comparison

Measured September 11, 2026 on an InstaCloud Intel Xeon 6975P-C instance with
four allocated CPUs, affinity `0-3`, and four inference threads. Both engines use
Whisper large-v3-turbo INT8 weights from the same original checkpoint. The native
encoder also uses INT8 activations. This is transcription-only, without diarization.

| Engine | WER | S | D | I | Failed requests | Total request time | Mean request time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whisper-turbo.c | 3.7158% | 25 | 9 | 0 | 0 / 50 | 1,060.630 s | 21.213 s |
| whisper.cpp Q8_0 | 3.9344% | 26 | 10 | 0 | 0 / 50 | 599.413 s | 11.988 s |

The denominator is **915 normalized reference words**. Corpus WER is
`100 × (S + D + I) / 915`, not the mean of per-utterance WERs. The two-word
difference is not evidence of general accuracy superiority. whisper.cpp completes
this slice 1.77× faster. The latency figures are one run per utterance on shared
CPUs, not repeated-trial medians.

## Corpus and scope

The [manifest](manifest.json) contains the exact IDs, reference transcripts,
durations, row indices and SHA-256 hashes of the original FLAC and decoded WAV
files. Selection is rows 0–49 of `openslr/librispeech_asr`, `clean/test`, at revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. Total audio is **328.2650625 seconds**;
all clips belong to **speaker 6930**.

This follows the selection procedure of a public 50-clip quantization study;
[published scores and their limitations](../../wer/README.md#dataset-and-published-results)
are documented separately. This is not the complete 2,620-utterance split, a
multilingual evaluation, a Chinese accuracy test, or a diarization benchmark.

[LibriSpeech](https://www.openslr.org/12) is licensed CC BY 4.0 and is attributed
to Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, using LibriVox
recordings. The reference text in these records retains that attribution/license.
The audio is not bundled here. Decoding FLAC to mono 16 kHz PCM16 WAV is lossless;
there is no resampling, cropping, amplitude normalization or output-based filtering.

## Runtime configuration

Native revision: `65a9d8126102df421ac62b157243c34f0aad6a66`.
whisper.cpp revision: `927cfce34f31707e17f2bff35c349632fb9e2c3a`.
Model hashes, source-archive hashes, compiler versions and scoring-source hashes
are in [provenance.json](provenance.json).

Both models stay resident. Engines run sequentially on the same instance, with
one discarded warm-up followed by 50 individual HTTP requests in manifest order.
Reference text is never supplied to inference. Requests specify English, JSON,
and initial temperature zero. Startup, model loading, audio preparation, WAV
hash checks, warm-up and scoring are excluded from request timings.

The native server retains its greedy/three-beam/temperature-recovery policy.
whisper.cpp uses beam size five, best-of five, temperature increment 0.2, its
default timestamp decoding, no cross-request context, and no external VAD. It is
built in Release mode with `GGML_NATIVE=ON`, GPU backends disabled, and flash
attention enabled. AVX-512/VNNI support is confirmed by the runtime.
Different decoder policies mean this is an engine comparison, not a measurement
of quantization error alone.

The C scorer applies the same Unicode lowercase/NFKD/punctuation-removal policy
to reference and hypothesis. It does not use Whisper's EnglishTextNormalizer;
number formatting and contraction expansion are not normalized away.
[Full scoring and reproduction instructions](../../wer/README.md#scoring).

## Audit files

- [Native raw responses](native.jsonl) and [per-utterance error counts](native-summary.json).
- [whisper.cpp raw responses](cpp.jsonl) and [per-utterance error counts](cpp-summary.json).
- Discarded warm-ups: [native](native-warmup.json), [whisper.cpp](cpp-warmup.json).

All audio hashes and utterance IDs were checked. Final C/libunistring rescoring
on Linux and macOS produces identical aggregate and per-utterance counts, also
checked against an independent edit-distance implementation. C unit/CLI tests,
ASan/UBSan, and static analysis pass. Failed responses would count as deletions;
incomplete runs are rejected by the scorer.

## Reproduce

Build the [C tools](../../wer/README.md#build-and-run), prepare the manifest's
50-row slice, and verify its hashes. Start each server separately with the
configuration above. For the native server, set `OMP_NUM_THREADS=4`,
`OMP_DYNAMIC=FALSE`, `WHISPER_ACTIVATIONS=int8`, and CPU affinity `0-3`.
For whisper.cpp, use `-t 4 -p 1 -bo 5 -bs 5 -l en -ng`, with the same affinity.

Run the client and scorer as documented in the benchmark README. Recorded
responses can also be rescored offline: copy `manifest.json` and the two `.jsonl`
files into a new directory, then run `wer-bench score DIRECTORY native` and
`wer-bench score DIRECTORY cpp`. No model, audio download, or Python is needed
for offline scoring.
