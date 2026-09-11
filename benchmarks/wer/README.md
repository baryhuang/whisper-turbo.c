# LibriSpeech WER benchmark

Compare resident transcription servers on the same LibriSpeech audio with a
C11 downloader, lossless FLAC-to-PCM16 conversion, HTTP client and WER scorer.
No Python is required. The external whisper.cpp comparator is the only C++
inference implementation; it is not a dependency of the native service.

## Dataset and published results

[LibriSpeech test-clean](https://www.openslr.org/12) is public English audiobook
speech with reference transcripts, licensed CC BY 4.0. The complete split has
2,620 utterances. whisper.cpp provides an
[upstream LibriSpeech evaluation harness](https://github.com/ggml-org/whisper.cpp/tree/927cfce34f31707e17f2bff35c349632fb9e2c3a/tests/librispeech).
Corpus attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev
Khudanpur; source recordings are from LibriVox.

The default 50-row selection follows the English selection procedure in
[in0vik's published quantization benchmark](https://gist.github.com/in0vik/8cd17a3bc51d88f740b8a7c190ea154f/33cd4773a9b35395a537d471098af476539928b8).
That author reports **4.04% WER for Turbo Q5_0** and **4.70% for small Q8_0**
on 50 test-clean clips. These are author-reported results on different models or
precision, not our measured baseline. The gist does not pin its whisper.cpp or
dataset revisions, so matching its selection procedure is not proof of an exact
historical reproduction. Its process-launch timings include model loading and
are not comparable with resident-request timings.

At dataset revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, rows 0–49 contain
**328.2650625 seconds, 915 normalized reference words, and one speaker (6930)**.
This is a reproducible small-subset comparison, not full-corpus WER or a
multispeaker/general-accuracy claim. No audio is selected or excluded based on
either engine's output. The exact IDs, references, sample counts and FLAC/WAV
SHA-256 hashes are in the [manifest](../results/librispeech-50/manifest.json).

## Scoring

Both reference and hypothesis use Unicode lowercase, NFKD decomposition, deletion
of characters other than letters/numbers/underscore/whitespace, and whitespace
collapse. Punctuation is deleted, not replaced: `don't` becomes `dont` and
`re-enter` becomes `reenter`. Numbers and contractions are not expanded. GNU
libunistring handles Unicode in C; this matches the published study's normalization operations
for the English text, excluding timestamp stripping because the JSON `text`
field has no presentation timestamps. Unicode-version differences can affect
other scripts or newly introduced characters.

This is **not Whisper's EnglishTextNormalizer**, which is used by the upstream
full-corpus harness. Scores from that harness or other leaderboards must not be
treated as interchangeable with this normalization.

The scorer computes sentence-local Levenshtein alignments and sums substitutions,
deletions and insertions over all reference words:

`corpus WER = 100 × (S + D + I) / N`

It does not average per-utterance percentages or allow alignments across utterance
boundaries. Alignment ties prefer substitution, deletion, then insertion; the
total edit distance is unaffected. HTTP/transport/malformed-response failures
are counted as empty hypotheses (all reference words deleted) and reported
separately. An incomplete run or an ID/reference mismatch is rejected, not scored
as if the remaining files did not exist. Raw response text is retained for audit.

## Resident comparison protocol

Use the same host, CPU affinity and thread count for both engines. Run only one
inference service at a time. Each loads its model once and transcribes the first
item as a discarded warm-up, then processes every selected item in manifest order
as a separate request. Each request explicitly selects English, JSON and initial
temperature zero. No prompt, external VAD, diarization or cross-request transcript
conditioning is used. WAV hashes are checked before each request.
Reference transcripts are used only for scoring and are never sent as prompts.

The native server uses its shipping greedy → three-beam → five-candidate
temperature-fallback policy. The whisper.cpp request explicitly selects beam size
five, best-of five, and temperature increment 0.2, retaining its default timestamp
decoding. These reflect different shipping decoders; this is an end-to-end engine
comparison, not an isolated quantization-error experiment or identical-logits
test. Both use Turbo INT8 weights; the native server also enables INT8 encoder
activations. Do not substitute a floating-point model.

Only HTTP request wall time is recorded. Startup, model loading, FLAC conversion,
audio hashing and warm-up are outside that interval. Do not compare these numbers
with benchmarks that reload the model for each file. This harness does not measure
memory caps, diarization accuracy or hosted API accuracy.

## Build and run

Benchmark-only C dependencies: libcurl (7.85+), cJSON, libsndfile, GNU libunistring
and OpenSSL. No C++ Unicode runtime is needed.
On Debian, install their development packages and `pkg-config`, then:

```sh
make -C benchmarks/wer all test
benchmarks/wer/wer-bench prepare /path/to/new-corpus 50
```

The root Makefile also exposes `make wer-tools check-wer`.

The downloader refuses an existing output directory. It checks the dataset's
current revision against the audio assets, saves the resolved revision, and
converts the original mono 16 kHz FLAC to PCM16 WAV without resampling. An existing
manifest and matching WAV files can be reused offline for inference and scoring.
To prepare the entire split, use count `2620`; an optional fourth argument is the
starting row offset. Larger runs must be reported with their actual manifest,
not as the default 50-clip result.

On Homebrew, pass
`UNICODE_CFLAGS="-I$(brew --prefix libunistring)/include"` and
`UNICODE_LIBS="-L$(brew --prefix libunistring)/lib -lunistring"` to `make`.

With the native server already running locally:

```sh
benchmarks/wer/wer-bench run /path/to/corpus \
  http://127.0.0.1:8080/v1/audio/transcriptions native native
benchmarks/wer/wer-bench score /path/to/corpus native
```

Stop the native server before starting the CPU-only whisper.cpp server with its
Q8_0 model and the same thread/affinity settings. Then:

```sh
benchmarks/wer/wer-bench run /path/to/corpus \
  http://127.0.0.1:8081/inference cpp cpp
benchmarks/wer/wer-bench score /path/to/corpus cpp
```

Results are `LABEL.jsonl`, `LABEL-warmup.json`, and `LABEL-summary.json` in the
corpus directory. Existing result files are not overwritten; choose another
label for another trial. The client accepts only loopback HTTP endpoints and
sends no credentials or audio to hosted inference APIs.
