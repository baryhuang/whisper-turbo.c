# Five-minute transcription validation

Native C HTTP transcription with INT8 Whisper weights and encoder activations,
on InstaCloud Intel Xeon 6975P-C with eight vCPUs/threads. Requests use
`model=gpt-4o-transcribe-diarize`, `response_format=diarized_json`, and
`chunking_strategy=auto`. Language is detected automatically.

| Recording | Audio duration | HTTP status | Request time | Transcript speaker labels |
| --- | ---: | ---: | ---: | ---: |
| Sample 1 | 300 s | 200 | 148.04 s | 1 |
| Sample 2 | 300 s | 200 | 218.23 s | 2 |

Each request processes ten ASR windows and one full-recording diarization pass.
Both responses contain nonempty text and finite, ordered, non-overlapping speaker
segments within the input duration. Concatenating segment text exactly reproduces
the complete transcript.

Times cover one request per recording after an excluded warm-up; they exclude
startup and are not repeated-trial medians. They are separate from the short-clip
CPU scaling and cost benchmark. The configured memory ceiling was 8,192 MiB;
the server's process RSS high-water mark was 1,012,293,632 bytes, not a service-wide
memory-cap certification.

The diarizer detects three speakers in Sample 2, while only two receive text
segments. A diarizer speaker count is not necessarily the number of speakers in
the ASR output. No transcription or diarization accuracy equivalence is claimed.

See [machine-readable results](runs.json). Private audio and transcript text are
not included in these records.
