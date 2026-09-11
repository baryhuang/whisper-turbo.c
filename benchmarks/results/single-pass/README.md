# Resident single-pass transcription

The HTTP endpoint and combined CLI use the same C inference and renderer:
one full-recording Whisper pass, captured cross-attention DTW alignment, one
Community-1 pass, then assignment of the existing transcript to speaker intervals.
There is no per-speaker ASR invocation. Long recordings use bounded 30-second ASR
windows, independent of speaker count.

## Measurement protocol

Intel Xeon 6975P-C, **4 vCPU, four threads**, GCC 12.2/OpenMP, INT8 weights and
encoder activations. No floating-point benchmark variant is included. Decoder
activations and the diarizer retain their existing floating-point operations.
The service has an 8,192 MiB memory ceiling, not a measured memory requirement.
The guest exposes five logical CPUs; both processes are pinned to CPUs 0–3.

Three measured direct calls: **35.308962, 35.237685, 35.273096 seconds**.
Three measured HTTP requests: **36.298806, 35.537225, 35.739450 seconds**.
Medians: **35.273096 seconds direct**, **35.739450 seconds HTTP**.
All HTTP responses matched the direct output byte-for-byte. Transcript text and
speaker segments also matched the earlier eight-thread result.
See [four-CPU records](runs-4cpu.json); [earlier eight-thread records](runs.json)
are retained separately.

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
OMP_NUM_THREADS=4 WHISPER_ACTIVATIONS=int8 taskset -c 0-3 ./build/resident-pipeline-bench \
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
