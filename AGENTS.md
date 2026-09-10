# Project scope

Work in `/Users/buryhuang/git/whisper-turbo.c`. Its origin is
`https://github.com/baryhuang/whisper-turbo.c.git`.

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
- Match the OpenAI transcription API contract; never manufacture timestamp or confidence data.
- The C importer and x86 INT8 inference kernels are implemented; HTTP API work remains.
- Keep W8A8 activation quantization opt-in until a wider accuracy corpus passes.
- Cloud memory counters are lifetime peaks: restart the isolated benchmark service
  before a fresh memory comparison, verify model cache eviction, and do not run
  overlapping exec commands on one service. Preserve failed/inconclusive runs.
