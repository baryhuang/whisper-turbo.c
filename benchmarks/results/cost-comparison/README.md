# Historical cost and crop-based transcription evidence

These API records describe the earlier per-speaker-crop implementation. They are
excluded from the current resident-service latency and cost comparison. See the
[single-pass resident results](../single-pass/README.md) for the shared CLI/HTTP pipeline.

September 10, 2026. See [methodology](../../../docs/cost-comparison.md) for
rate-card sources, assumptions, exclusions and limitations.

- `diarized-http.json`: one cold-model native C HTTP transcription with
  Community-1 diarization, eight OpenMP threads, opt-in INT8 encoder activations,
  Intel Xeon 6975P-C. Peak memory is a fresh cgroup lifetime counter, not sampled
  RSS or configured memory. VM startup is outside the measured window.
- `validation.json`: build identities and additional API correctness checks.

The HTTP workload used `model=gpt-4o-transcribe-diarize`,
`response_format=diarized_json`, `language=en`, and a 27.27-second PCM16/16kHz WAV.
Its private audio and full transcript are deliberately excluded. Three segments
were nonempty, ordered, within input duration and consistently labeled `A`.
The observer includes server startup, health polling, curl, inference and graceful
shutdown. Five model files had zero initially resident pages; no swap was present.

The separate A–B–A fixture uses two speakers and repeats the first speaker.
An eight-second reference from the first appearance tests known-speaker matching.
The SSE response must preserve `agent` across both appearances, assign the other
speaker `A`, use matching segment IDs, and finish with the complete joined text.
This checks identity consistency, not diarization error rate or transcription
accuracy. SSE is buffered, not incremental inference. Follow-up runs have no
fresh memory claim and do not replace the cost-table workload.

The standalone C diarization CLI also passed `--check-aba` after the reusable
pipeline extraction. Local `make all server diarization check` passed with
`-Wall -Wextra -Wpedantic -Werror`; HTTP tests passed ASan/UBSan and TSan.
The Linux server and HTTP tests built with GCC 12.2/OpenMP and `-Werror`.
The Docker recipe was not built. No hosted OpenAI or AssemblyAI request was made.

## Excluded attempts

An initial macOS real-model request exposed an undersized worker stack. The
fixed implementation uses a two-MiB worker stack and a heap audio window.
A subsequent ASan real-model cancellation check returned HTTP 504 without a
sanitizer error; cancellation takes effect at numerical-work boundaries.
A full scalar macOS request was manually cancelled without a completed result.
Neither is a completed timing or memory benchmark. An initial cloud SSE launch
failed shell quoting before inference; the corrected launch completed successfully.
