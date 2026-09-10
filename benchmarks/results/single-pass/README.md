# Resident single-pass transcription

The HTTP endpoint and combined CLI use the same C inference and renderer:
one full-recording Whisper pass, captured cross-attention DTW alignment, one
Community-1 pass, then assignment of the existing transcript to speaker intervals.
There is no per-speaker ASR invocation. Long recordings use bounded 30-second ASR
windows, independent of speaker count.

## Measurement protocol

Intel Xeon 6975P-C, eight threads, GCC 12.2/OpenMP, INT8 encoder activations.
Three measured direct calls: **19.515750, 19.547507, 19.460840 seconds**.
Three measured HTTP requests: **19.238235, 19.114012, 19.255878 seconds**.
Medians: **19.515750 seconds direct**, **19.238235 seconds HTTP**.
All HTTP responses matched the direct output byte-for-byte. The complete text
also matched an ASR-only request. See [machine-readable records](runs.json).

The service stays running with the Whisper model resident. Discard one complete
warm-up request, then measure three sequential requests of the same 27.27-second
PCM16/16kHz recording. HTTP time covers upload, inference and response receipt.
Startup, model initialization, warm-up and shutdown are excluded from reported
latency and cost. No model-cache eviction or VM restart occurs between requests.

The direct benchmark also loads the model once and discards its first request.
It measures three calls to the same `wt_transcribe` and `wt_render` functions,
including JSON rendering. Its C checks require identical responses across all
four calls. Compare its output with the HTTP response byte-for-byte.

```sh
make build/resident-pipeline-bench OPENMP=-fopenmp
OMP_NUM_THREADS=8 WHISPER_ACTIVATIONS=int8 ./build/resident-pipeline-bench \
  turbo-q8.whtrbo /path/to/community-1 speech.wav en
```

The first `RESIDENT` record has `warmup=true`; it is excluded, not averaged into
the measured trials. Each measured short-clip request must report `asr_windows=1`
and `diarization_passes=1`. Full JSON goes to stdout; timing records go to stderr.
Private recordings and transcript text are excluded from public records.

An A–B–A fixture separately checks repeated-speaker identity and preservation of
the original full-recording ASR text. This is not a DER/WER benchmark or a claim
of alignment parity with upstream Whisper/pyannote. Whitespace-based grouping
remains coarse for unspaced languages.

Cost scenarios describe active request processing, not an always-on monthly bill.
Resident memory continues to cost money between requests. Sustained throughput,
idle utilization, network, storage and account fees must be accounted for separately.

## Excluded observations

Earlier startup-inclusive observations belong to a different measurement protocol
and are excluded from the resident timing/cost comparison. A first resident run
completed but its command gateway returned 502 and lost timing output; its JSON
was recovered, but no timing from that run is used. Repeated tests persist their
results on the benchmark volume independently of the command connection.
