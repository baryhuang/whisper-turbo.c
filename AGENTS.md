# Project scope

Work in `/Users/buryhuang/git/whisper-turbo.c`. Its origin is
`https://github.com/baryhuang/whisper-turbo.c.git`.

- Implement Whisper Turbo in C, including model preparation, HTTP serving, and tests.
- No Python or C++ implementation or runtime/build dependencies.
- Do not import A113X sources or target that board.
- Inspect InstaCloud CPU architecture and resource limits before selecting CPU optimizations.
- Preserve upstream source attribution and describe unverified behavior accurately.
- Match the OpenAI transcription API contract; never manufacture timestamp or confidence data.
- The extraction currently builds only the inherited CLI and encoder benchmark. HTTP API work remains.
