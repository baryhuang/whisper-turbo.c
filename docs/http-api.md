# Transcription HTTP API

`whisper-turbo-server` is a native C11 HTTP server with one resident Whisper
large-v3-turbo model and one active transcription. It exposes
`POST /v1/audio/transcriptions` using multipart requests and JSON/text/diarized
responses. No Python, C++ runtime, or external inference executable is involved.
Whisper stays resident; optional diarizer checkpoints are opened per request and
closed after the request's single diarization pass. HTTP and the combined CLI
use the same inference function and response renderer.

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
| `file` | Required file part; mono PCM16 WAV at 16 kHz, up to 300 seconds. |
| `model` | Required: `whisper-large-v3-turbo`, `whisper-1`, or `gpt-4o-transcribe-diarize`. All are local model aliases; the last enables Community-1. |
| `language` | Optional lowercase model language code. Omission detects the language from the first non-silent window. |
| `response_format` | `json` (default), `text`, or `diarized_json` for the diarization model. |
| `temperature` | Zero or omitted: starts with greedy decoding, with bounded internal quality-gated fallback. Other initial temperatures are rejected. |
| `stream` | Diarization model: `true` returns buffered SSE events. Whisper aliases: ignored, as on the hosted Whisper route. |
| `chunking_strategy` | Diarization model: `auto`, required for audio longer than 30 seconds. Custom server-VAD settings are rejected. |
| `known_speaker_names[]`, `known_speaker_references[]` | Up to four paired names and 2–10-second PCM16/16kHz WAV base64 data URLs; diarization model only. |
| `prompt` | Empty or omitted only. Nonempty conditioning prompts return an explicit error. |
| Word timestamps, SRT, VTT, verbose JSON, logprobs | Not implemented; requests return an explicit error. Diarized segment timestamps are supported. |
| Compressed audio, stereo, other sample rates | Not implemented; rejected rather than misread or silently converted. |

`whisper-1` is a compatibility **alias**, not a claim that the server runs
OpenAI's hosted Whisper model. `gpt-4o-transcribe-diarize` is also an interface
alias, not OpenAI's proprietary model; other GPT model names are rejected.
Unknown fields and duplicate scalar fields are rejected; options are never silently
dropped except the documented Whisper `stream` behavior.

Accepted recordings are processed in successive 30-second windows with bounded
workspace. All accepted audio is processed, including speech after 30 seconds.
Window text is joined without cross-window prompt conditioning or timestamp-based
overlap correction; speech crossing a boundary can lose continuity. Exact digital
silence is skipped. See [decoding and recovery](#decoding-and-recovery) for
repetition, low-confidence and decoder-context handling.

## Decoding and recovery

Every window first uses greedy decoding. A completed candidate is rejected when
its mean token log probability is below −1.0 or, for sequences longer than 32
tokens, its last-32-token entropy is below 2.4. Entropy uses natural logarithms
and includes the end-of-text token. A high no-speech probability (>0.6) combined
with low confidence identifies a silent window; high no-speech probability alone
does not discard confident text. No-speech probability is measured before token
filtering.

Rejected or context-exhausted output is retried with three quality-gated beams,
then temperatures 0.2, 0.4, 0.6, 0.8 and 1.0 as needed. Each positive temperature
evaluates five candidates sequentially and selects the highest-scoring candidate
that passes the same checks. Sampling scores use the temperature-adjusted,
filtered distribution. Fixed request-local seeds make retries reproducible on
the same runtime; they do not guarantee identical output across hardware.

Retries reuse the encoded audio and cross-attention cache. They do not switch
precision, force another language, split the recording into shorter requests,
or repeat diarization. Difficult audio can take longer because of decoder retries.
Every accepted hypothesis must generate its own end-of-text token within the
448-position context. If no completed candidate passes, the entire request
returns `422` with code `decoding_failed`; partial text is not returned.

These safeguards follow the [whisper.cpp decoding criteria](https://github.com/ggml-org/whisper.cpp/blob/927cfce34f31707e17f2bff35c349632fb9e2c3a/src/whisper.cpp).
They are heuristics, not proof of recognition accuracy or absence of hallucinations.
The HTTP API and combined diarized CLI use this policy; the basic encoder/decoder
CLI retains its separate English greedy path.

## Diarized transcription

Set `WHISPER_DIARIZATION_MODELS` to a directory containing the four
[Community-1 checkpoint files](diarization.md), then request:

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=gpt-4o-transcribe-diarize \
  -F response_format=diarized_json -F chunking_strategy=auto
```

The response implements the `TranscriptionDiarized` shape:

```json
{
  "task": "transcribe",
  "duration": 6.0,
  "text": "Hello. Hi.",
  "segments": [
    {"type":"transcript.text.segment","id":"seg_001","start":0.1,"end":2.0,"speaker":"A","text":"Hello."},
    {"type":"transcript.text.segment","id":"seg_002","start":3.0,"end":5.8,"speaker":"B","text":"Hi."}
  ],
  "usage": {"type":"duration","seconds":6.0}
}
```

This is an illustrative schema example, not a benchmark transcript. `json` returns
`text` and duration `usage`, without annotations. `text` returns only combined text.
Usage reports actual input duration; it is not an OpenAI bill or token count.

Community-1 diarizes the complete recording once before Whisper's single
full-recording ASR pass. In diarized mode, 30-second ASR windows without acoustic
speech activity are skipped, with 250 ms padding around detected speech. The
original windows and timestamps are retained; audio is not concatenated or
transcribed per speaker. Overlap-aggregated segmentation, rather than an isolated
positive mask, decides whether a recording contains speech. Activity gaps up to
100 ms are merged and bursts shorter than 250 ms are rejected, following the
default speech-duration settings documented by whisper.cpp. Very brief isolated
utterances can therefore be omitted. If real speech is too short for the usual
two-second clean-embedding filter, finite nonzero embeddings supported by accepted
speech activity are clustered instead; missing evidence still returns an error.
This reduces
silence hallucinations but does not guarantee recognition accuracy inside an
active window. The stages run sequentially to bound peak memory.
Six selected cross-attention heads are captured during accepted greedy decoding.
After recovery, the winning token sequence is replayed through the cached decoder
to capture its attention, without generating a new transcript or rerunning the encoder.
Normalization, median filtering and dynamic time warping provide monotonic text
alignment without a second transcription pass. One cached decoder consume of the
end token supplies the alignment sentinel; it does not generate another transcript.
These are model-derived estimates, not uniformly distributed or invented times.

Subword pieces are grouped at whitespace boundaries. Each group is assigned to
the exclusive speaker interval with the largest time overlap; ties and groups in
gaps use the nearest interval. Before speaker assignment, aligned word groups
with no overlap with padded acoustic speech activity are discarded. A group is
also discarded when more than two seconds of its span lack acoustic support and
speech covers less than half the span. Padded intervals are unioned, not counted
twice. This duration/acoustic heuristic is inspired by Whisper's
[long-word/silence anomaly checks](https://github.com/openai/whisper/blob/main/whisper/transcribe.py),
not a claim of upstream parity. It prevents
silence hallucinations from borrowing a speaker label from distant real speech;
it can also omit words when acoustic detection or alignment is wrong. Retained
text and model-derived timestamps are unchanged. Consecutive groups from one
speaker form a segment.
Empty ASR results do not produce segments. Speaker
labels are assigned in first emitted appearance order, retaining identity across
later turns. Segment IDs are unique within a response. Concatenating segment texts
exactly reconstructs the speech-supported text, including whitespace. Segments after
the first can begin with a space. Text is not regenerated for individual speaker
intervals. No synthetic confidence scores or token usage are generated.

Up to four known-speaker references are accepted as paired repeated fields:

```text
known_speaker_names[]=agent
known_speaker_references[]=data:audio/wav;base64,...
```

Names must be unique, nonempty and at most 63 UTF-8 bytes with no control
characters. References must be canonical base64 WAV data URLs, not remote URLs;
they are decoded in memory without network fetches or filesystem paths. The C
segmentation/embedding model extracts the dominant clean voice and compares it to
diarization centroids. Matching is one-to-one, requires cosine similarity above
0.65 and a 0.05 margin over the next cluster; unmatched/ambiguous clusters stay
anonymous. These local thresholds are experimental, not parity with OpenAI or
speaker-identity certification. Reference names such as `A` cannot collide with
generated anonymous labels.

`stream=true` returns `text/event-stream`: text delta events, segment events for
`diarized_json`, then one text done event. Deltas carry matching `segment_id`s for
diarized output. **Events are buffered until the entire request completes**, not
emitted token-by-token during inference. Done events omit optional token usage
because this runtime cannot report OpenAI token billing. Errors before completion
remain ordinary HTTP JSON errors. The five-second response-write deadline applies
to the complete buffered event sequence.

Alignment and diarization can misplace a speaker boundary, especially across long
pauses or unclear speech. Whitespace grouping can produce coarse segments in
languages written without spaces; multilingual word-boundary quality is not
validated. No external acoustic forced-aligner model is run. The
exclusive timeline chooses one speaker during overlap; it does not isolate
simultaneous voices. No DER/WER equivalence to OpenAI or upstream pyannote is claimed.
An absent model directory returns 503; diarization failure returns an error, not
made-up single-speaker output. These features do not implement every option in
the wider Audio API. Speech generation, translations, voices and voice-consent
endpoints are not provided.

## Resource and security limits

- One resident memory-mapped Whisper model and one persistent inference worker,
  with a fixed two-MiB stack. Diarizer workspaces are request-scoped.
- One active upload/transcription/response; additional transcription requests get
  `429` and `Retry-After: 1`. No pending transcription queue is retained.
- At most eight accepted connections; excess connections get `503`.
- A 25,000,000-byte multipart-body limit, 8,192-byte HTTP-header limit,
  2,048-byte per-part header limit, and at most 32 multipart parts.
- Up to 300 seconds of audio and 262,144 output text bytes. Upload data stays in
  the bounded request buffer; filenames are never opened or used as filesystem paths.
- At most 512 emitted speaker segments and four references, each up to 430,000
  encoded bytes within the total upload cap. Serialized JSON/SSE is also bounded.
- Absolute deadlines: ten seconds for headers, 30 seconds for the upload,
  300 seconds for inference by default, and five seconds for response writes.
- Client disconnects and shutdown cancel inference at diarization-chunk, encoder-layer and decoder-token
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
{"error":{"message":"Use json, text, or diarized_json.","type":"invalid_request_error","param":"response_format","code":"unsupported_parameter"}}
```

Typical status codes are `400` for invalid/unsupported input, `401` for invalid
authentication, `408` for upload timeout, `413` for size/duration limits, `422`
for exhausted decoder/output capacity, `429` for a busy worker, and `504` for
inference cancellation/deadline. Errors do not unload the model; subsequent
requests can continue.

## Validation

Two five-minute PCM16 recordings passed complete diarized HTTP requests with
automatic language detection and INT8 encoder activations on InstaCloud. Both
processed all ten ASR windows and returned ordered speaker segments whose text
concatenates exactly to the full transcript. See [five-minute validation](../benchmarks/results/long-audio/README.md)
for results and accuracy limitations.

On InstaCloud's Intel Xeon 6975P-C, with eight threads, the resident server passed
real-model multipart requests, repeated-transcript checks, automatic English
language detection, text/JSON output, live health checks, and overload rejection.
A 42-second fixture with all speech after 30 seconds was transcribed successfully.
The 120-second repeated-speech and near-25-MB upload tests also passed.

The earlier ASR-only test-run maximum was **1,029,955,584 bytes cgroup memory** and
**1,012,088,832 bytes process RSS**. These cover the resident server, HTTP handling,
model cache, inference, and the C test client in the same execution cgroup. No
configured swap was present, and model-cache residency was checked before startup.
They are observed peaks, not proof of a platform-enforced 1.5-GB memory cap.

The single-pass diarized endpoint was measured as an already-running service:
one discarded warm-up followed by three requests. Its median was **19.238235
seconds** for 27.27 seconds of audio. The transport-equivalent resident direct
pipeline median was **19.515750 seconds**. Startup, model initialization and
warm-up are excluded from both. All measured responses matched byte-for-byte;
the diarized transcript text also matched the ASR-only endpoint exactly.

The 26-second A–B–A fixture retained labels `A`, `B`, `A`. With an eight-second
reference, SSE returned `agent`, `A`, `agent` in **18.693373 seconds**, preserving
the complete ASR text and whitespace across delta events. This is a functional
check, not part of the three-run cost median or a DER/WER evaluation.
Before the long-audio check, the server process lifetime RSS high-water mark across warm-up, repeated requests
and these checks was **1,001,713,664 bytes**. This is process RSS, not a new cgroup
peak-memory acceptance result; it cannot certify the total 1.5-GB service target.
See [resident measurements](../benchmarks/results/single-pass/README.md) and
[cost assumptions](cost-comparison.md).

A 42-second repeated-speech request with `chunking_strategy=auto` also passed,
including transcript text after 30 seconds and exact segment-text concatenation.
This functional check is excluded from the short-clip latency/cost median.

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

The transport-equivalent CLI calls the same `wt_transcribe` and `wt_render` code:

```sh
make diarized-cli OPENMP=-fopenmp
OMP_NUM_THREADS=8 ./build/whisper-turbo-diarize \
  turbo-q8.whtrbo /path/to/community-1 speech.wav en
```

Its default output is `diarized_json`, `chunking_strategy=auto`; its optional last
argument selects language. The CLI has no reference-speaker argument; HTTP retains
reference fields. Both use identical inference for equivalent options.

Without OpenMP, omit `OPENMP=-fopenmp`; the server uses the portable single-threaded
path on non-x86 platforms. The command is:

```text
whisper-turbo-server MODEL.whtrbo [PORT [BIND_IPV4]]
```

`OMP_NUM_THREADS` accepts 1–8. `WHISPER_REQUEST_TIMEOUT` accepts 1–3600 seconds.
For diarization, set `WHISPER_DIARIZATION_MODELS=/path/to/community1/models`;
the directory must contain all four checkpoint files described in
[model setup](diarization.md). Mount this directory read-only in a container and
set the environment variable to its container path.
Set `WHISPER_API_KEY` in the environment to enable bearer authentication; do not
put credentials in source or command-line arguments. Existing SIMD controls and
opt-in `WHISPER_ACTIVATIONS=int8` also apply. FP32 activations remain the default.

Set `WHISPER_DIAGNOSTICS=1` for content-free ASR window, alignment, and diarization
diagnostics on stderr. Leave it unset for normal operation; diagnostics do not
include transcript text or uploaded audio.

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
and recovery. It also checks diarized JSON/SSE HTTP responses, repeated multipart
reference fields, canonical base64 reference decoding, segment bounds, and SSE
text assembly. The production executable has no mock mode.

`http-model-test` requires an already-running, unauthenticated loopback server and
the JFK fixture. It checks real transcripts without printing their contents,
warm-request repeatability, automatic language detection, live health checks,
overload rejection, text output, and speech placed after 30 seconds. Its generated
long-audio fixture exists only in memory.

Optional `smoke`, `repeat`, and `limits` arguments select one real request, ten
repeated requests, or the 300-second/near-upload-limit tests respectively. The
near-limit fixture adds a WAV metadata chunk; its actual speech duration is still
11 seconds. All test clients are written in C.
