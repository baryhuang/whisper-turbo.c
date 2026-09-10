# Project scope

Work in `/Users/buryhuang/git/whisper-turbo.c`. Its origin is
`https://github.com/baryhuang/whisper-turbo.c.git`.

- Work directly on `main`; do not create new branches unless explicitly requested.
- Implement Whisper Turbo in C, including model preparation, HTTP serving, and tests.
- No Python or C++ implementation or runtime/build dependencies.
- Do not import A113X sources or target that board.
- The two deployment targets are `amd-epyc-avx2` and `intel-xeon-avx512`;
  see `docs/service-targets.md` and `targets/` for their contracts.
- The complete API service must peak strictly below 1,500,000,000 bytes
  (decimal 1.5 GB), including model loading, resident weights, request handling,
  inference, all child processes, and service-accounted file cache. This is
  a requirement, not a measured result. Do not substitute 1.5 GiB or model size.
- Start with one resident model, one active inference request and eight threads;
  bound uploads, queued requests, long-audio processing, and output buffers.
- Inspect InstaCloud CPU architecture and resource limits before selecting CPU optimizations.
- Preserve upstream source attribution and describe unverified behavior accurately.
- Keep READMEs end-user-facing: document available features, usage, limitations,
  and verified final results, not collaboration history, internal workflow, or plans.
- Match the OpenAI transcription API contract; never manufacture timestamp or confidence data.
- The C importer, x86 INT8 inference kernels, and resident HTTP transcription server
  are implemented. docs/http-api.md records the supported subset; full compatibility
  (codecs, prompts, sampling, and real timestamps/diarized transcripts) remains incomplete.
- The optional Community-1 diarizer in `src/diarization/` is experimental and
  standalone. Upstream parity, overlap accuracy, and resident HTTP-service memory
  are unverified. Sequential CLI memory is measured in docs/instacloud-diarization.md.
  Keep model files, credentials, and private test audio untracked.
- Keep W8A8 activation quantization opt-in until a wider accuracy corpus passes.
- Cloud memory counters are lifetime peaks: restart the isolated benchmark service
  before a fresh memory comparison, verify model cache eviction, and do not run
  overlapping exec commands on one service. Preserve failed/inconclusive runs.
