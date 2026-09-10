# Transcription HTTP API

`whisper-turbo-server` is a native C11 HTTP server with one resident Whisper
large-v3-turbo model and one active transcription. It exposes
`POST /v1/audio/transcriptions` using multipart requests and JSON/text responses.
No Python, C++ runtime, external inference executable, or per-request model load
is involved.

The endpoint follows the supported subset of the
[OpenAI transcription interface](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create).
**It is not yet a complete replacement for every Whisper API option.**

## Send a request

With the server listening on port 8080:

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=whisper-1 \
  -F language=en
```

The JSON response contains only the recognized text:

```json
{"text":"Recognized speech appears here."}
```

Use `-F response_format=text` for a plain-text response. Clients with a configurable
OpenAI base URL can use `http://127.0.0.1:8080/v1` and `model="whisper-1"` for this
supported request subset. When authentication is enabled, add
`-H "Authorization: Bearer $WHISPER_API_KEY"`.

## Compatibility

| Field or behavior | Implemented behavior |
| --- | --- |
| `file` | Required file part; mono PCM16 WAV at 16 kHz, up to 120 seconds. |
| `model` | Required: `whisper-large-v3-turbo` or `whisper-1`. Both select the locally loaded Turbo model. |
| `language` | Optional lowercase model language code. Omission detects the language from the first non-silent window. |
| `response_format` | `json` (default) or `text`. |
| `temperature` | Zero only; deterministic greedy decoding without temperature fallback. |
| `stream` | Boolean accepted; ignored for this Whisper endpoint, which returns a non-streaming response. |
| `prompt` | Empty or omitted only. Nonempty conditioning prompts return an explicit error. |
| Timestamps, SRT, VTT, verbose/diarized JSON | Not implemented; requests return an explicit error. |
| Compressed audio, stereo, other sample rates | Not implemented; rejected rather than misread or silently converted. |

`whisper-1` is a compatibility **alias**, not a claim that the server runs
OpenAI's hosted Whisper model. GPT transcription model names are rejected.
Unknown fields and duplicate fields are rejected; options are never silently
dropped except the documented Whisper `stream` behavior.

Accepted recordings are processed in successive 30-second windows with bounded
workspace. All accepted audio is processed, including speech after 30 seconds.
Window text is joined without cross-window prompt conditioning or timestamp-based
overlap correction; speech crossing a boundary can lose continuity. Exact digital
silence is skipped. If decoding exhausts its token context, the entire request
fails with `422` instead of returning a silently truncated transcript.

The separate Community-1 executable is not integrated into this endpoint.
Speaker-attributed transcript segments and word alignment remain unsupported.

## Resource and security limits

- One resident memory-mapped model and one persistent inference worker.
- One active upload/transcription/response; additional transcription requests get
  `429` and `Retry-After: 1`. No pending transcription queue is retained.
- At most eight accepted connections; excess connections get `503`.
- A 25,000,000-byte multipart-body limit, 8,192-byte HTTP-header limit,
  2,048-byte per-part header limit, and at most 32 multipart parts.
- Up to 120 seconds of audio and 262,144 output text bytes. Upload data stays in
  the bounded request buffer; filenames are never opened or used as filesystem paths.
- Absolute deadlines: ten seconds for headers, 30 seconds for the upload,
  300 seconds for inference by default, and five seconds for response writes.
- Client disconnects and shutdown cancel inference at encoder-layer and decoder-token
  boundaries. In-flight numerical operations are not forcibly interrupted.
- HTTP/1.1 with `Content-Length`, optional `Expect: 100-continue`, and one request
  per connection. Transfer encodings, including chunked uploads, are rejected.
- Bind defaults to `127.0.0.1`. Non-loopback binding requires `WHISPER_API_KEY`.
  Remote access requires a TLS reverse proxy; the executable does not terminate TLS.
- `/health` is unauthenticated and responds while inference is active. The server
  does not log audio, transcripts, request bodies, or bearer credentials.

The model is mapped once at startup and reused. Passing HTTP tests does not by
itself certify the project's complete-service memory ceiling or broad transcription
accuracy. Multi-language quality, compressed codecs, alignment, and sustained
production-load behavior require further validation.

## Errors

Errors use an OpenAI-style JSON envelope:

```json
{"error":{"message":"Only json and text response formats are implemented.","type":"invalid_request_error","param":"response_format","code":"unsupported_parameter"}}
```

Typical status codes are `400` for invalid/unsupported input, `401` for invalid
authentication, `408` for upload timeout, `413` for size/duration limits, `422`
for exhausted decoder/output capacity, `429` for a busy worker, and `504` for
inference cancellation/deadline. Errors do not unload the model; subsequent
requests can continue.

## Validation

On InstaCloud's Intel Xeon 6975P-C, with eight threads, the resident server passed
real-model multipart requests, repeated-transcript checks, automatic English
language detection, text/JSON output, live health checks, and overload rejection.
A 42-second fixture with all speech after 30 seconds was transcribed successfully.
The 120-second repeated-speech and near-25-MB upload tests also passed.

The maximum completed test-run peak was **1,029,955,584 bytes cgroup memory** and
**1,012,088,832 bytes process RSS**. These cover the resident server, HTTP handling,
model cache, inference, and the C test client in the same execution cgroup. No
configured swap was present, and model-cache residency was checked before startup.
They are observed peaks, not proof of a platform-enforced 1.5-GB memory cap.

Tests pass with GCC 12.2/OpenMP on Linux and Apple Clang on macOS. The deterministic
HTTP suite also passes AddressSanitizer, UndefinedBehaviorSanitizer, and
ThreadSanitizer locally. The container recipe itself has not been built as part
of these tests.

See [request timings, raw records, and methodology](../benchmarks/results/instacloud-http/README.md).
The incomplete ten-request cloud run is excluded from memory acceptance; the
completed multi-request and resource-limit runs are reported separately.

## Build and run

```sh
make server OPENMP=-fopenmp
OMP_NUM_THREADS=8 ./build/whisper-turbo-server turbo-q8.whtrbo 8080
```

Without OpenMP, omit `OPENMP=-fopenmp`; the server uses the portable single-threaded
path on non-x86 platforms. The command is:

```text
whisper-turbo-server MODEL.whtrbo [PORT [BIND_IPV4]]
```

`OMP_NUM_THREADS` accepts 1–8. `WHISPER_REQUEST_TIMEOUT` accepts 1–3600 seconds.
Set `WHISPER_API_KEY` in the environment to enable bearer authentication; do not
put credentials in source or command-line arguments. Existing SIMD controls and
opt-in `WHISPER_ACTIVATIONS=int8` also apply. FP32 activations remain the default.

`Dockerfile.server` is the C-only container build recipe; the root `Dockerfile`
is a separate benchmark image that includes whisper.cpp. Mount the model read-only
at `/models/turbo-q8.whtrbo`, readable by UID 65534, and supply `WHISPER_API_KEY`
through your deployment's secret environment. The server container runs as UID
65534 and requires authentication because it binds to `0.0.0.0`.

## Tests

```sh
make check-http
make build/http-model-test
./build/http-model-test 8080 /path/to/jfk.wav
```

`check-http` uses a deterministic C test backend, not model inference. It exercises
multipart framing, binary uploads, malformed input, JSON escaping, authentication,
repeated requests, overload, upload/inference deadlines, disconnect cancellation,
and recovery. The production executable has no mock mode.

`http-model-test` requires an already-running, unauthenticated loopback server and
the JFK fixture. It checks real transcripts without printing their contents,
warm-request repeatability, automatic language detection, live health checks,
overload rejection, text output, and speech placed after 30 seconds. Its generated
long-audio fixture exists only in memory.

Optional `smoke`, `repeat`, and `limits` arguments select one real request, ten
repeated requests, or the 120-second/near-upload-limit tests respectively. The
near-limit fixture adds a WAV metadata chunk; its actual speech duration is still
11 seconds. All test clients are written in C.
