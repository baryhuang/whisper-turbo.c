# Resident C HTTP server validation

InstaCloud `whisper-bench`, existing branch `x86-q4`, service `kernel-bench`,
`us-east`; September 10, 2026. Intel Xeon 6975P-C, eight inference threads,
eight shared vCPUs / 8192 MiB allocation. The guest exposed nine logical CPUs,
AVX2/FMA and AVX-512/VNNI. The API was bound only to guest loopback port 8081.
No public transcription service was deployed.

## Successful requests

| Configuration / workload | HTTP time, seconds | Recording duration, seconds |
| --- | ---: | ---: |
| Default FP32 activations, JFK, cold weights | 73.232722 | 11 |
| INT8 encoder activations, JFK, cold weights | 14.347850 | 11 |
| Same resident INT8-activation server, second JFK request | 10.762503 | 11 |
| Same server, automatic language detection | 10.490872 | 11 |
| Same server, text output with speech only after 30 seconds | 10.506196 | 42 |
| INT8 encoder activations, repeated speech | 44.759023 | 120 |
| Same server, 24,999,000-byte WAV with metadata padding | 10.730716 | 11 |

These are individual measurements, not medians or controlled cross-precision
speed comparisons. INT8 encoder activations remain opt-in. The default preserves
FP32 activations; both configurations use the same 847,660,800-byte INT8-weight model.
The large upload's size comes from a WAV `JUNK` chunk, not additional audio.
The 120-second recording repeats the JFK fixture, so it is a resource/continuity
test, not an independent accuracy corpus.

## Completed execution peaks

| Execution | Cgroup peak, bytes | Maximum child process RSS, bytes |
| --- | ---: | ---: |
| Default-activation single request | 954,564,608 | 937,836,544 |
| INT8-activation single request | 954,466,304 | 937,738,240 |
| INT8-activation four-request suite | 1,002,192,896 | 983,998,464 |
| INT8-activation resource-limit suite | 1,029,955,584 | 1,012,088,832 |

The full HTTP server and C client run under one observer; the model remains loaded
across requests within each execution. Counters include charged file cache and
descendant processes, not just inference scratch. Client-side buffers are included
too, so these are conservative relative to the server alone. The instance is
restarted before each independent memory run; all accepted observations report
zero resident model pages before startup and no configured swap. No additional
remote execution overlaps a measured run.

HTTP latency is measured by the C client around upload and response. The first
request includes cold model-page loading, but not server process startup. The
observer's total duration additionally includes startup, the entire test sequence,
and shutdown. Successful client checks do not print transcript contents.

Raw records: [default request](resident-default-smoke.json),
[INT8-activation request](resident-smoke.json), [four-request suite](resident-suite.json),
[resource-limit suite](resident-limits.json).

## Build and correctness evidence

GCC 12.2.0, `-O3 -std=c11 -Wall -Wextra -Wpedantic -Werror -fopenmp`, pthreads,
libm, and libc. The server has no Python, C++ library, zlib, or external inference
process dependency. It uses the existing runtime-dispatched x86 kernels.
The exact source and server binary hashes for the resource-limit run are in
[release-validation.txt](release-validation.txt). The earlier short-request suite
predates the lock-free atomic shutdown flag; its inference implementation is the same.

The C HTTP suite verifies multipart boundaries, duplicate fields, malformed and
truncated WAVs, JSON escaping, authentication, request/connection limits,
`100-continue`, overload rejection, deadlines, disconnect cancellation, and
recovery. Linux warning-as-error builds and local ASan/UBSan/TSan checks pass.
The 120-second content check found 18 occurrences of a known word from the fixture,
versus two in the original 11-second clip; it is not a word-error-rate calculation.

## Excluded records

- [Language-lookup failure](resident-default.json): rejected before inference; not a performance pass.
- [Initial 120-second assertion](resident-limits-initial.json): HTTP succeeded but an arbitrary
  1,000-character output expectation failed. The content-based check is reported
  in the subsequent resource-limit record (954 text bytes, 18 expected-word occurrences).
- [Default multi-request transport failure](inconclusive-default-suite.json): platform HTTP 502,
  no returned complete observation; excluded.
- [Ten-request transport failure](inconclusive-repeat.json): seven successful response records
  were [recovered](recovered-repeat.json), but no final suite verdict or memory observation;
  not a successful ten-request or memory test.

These results establish the documented WAV/JSON/text HTTP subset on this Xeon.
They do not establish complete OpenAI interface compatibility, codec support,
word/speaker alignment, multilingual accuracy, AMD performance, a hard memory cap,
or sustained production-load acceptance. Build and request examples are in
[the API documentation](../../../docs/http-api.md).
