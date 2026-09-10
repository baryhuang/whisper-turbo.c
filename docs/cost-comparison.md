# ASR and diarization cost methodology

Prices checked September 10, 2026; USD. The native comparison uses an always-on,
already-warmed service. **Startup, model initialization, warm-up and shutdown are
excluded from both reported latency and the active-request cost calculation.**

## Equivalent work

The HTTP endpoint and combined C CLI both call `wt_transcribe` and `wt_render`.
One Whisper pass processes the full recording in bounded windows, capturing
cross-attention alignment during decoding. One Community-1 pass supplies speaker
intervals. Alignment assigns the existing transcript to those intervals; speaker
turns do not trigger additional Whisper passes. Stages run sequentially to bound
workspace, not concurrently with duplicate models.

The direct resident benchmark loads the model once, discards one full request,
and times three subsequent calls including JSON rendering. The HTTP benchmark
starts one server, discards one warm-up request, and times three requests including
upload and complete response receipt. Both use the same 27.27-second PCM16/16kHz
recording, English decoding, eight threads and opt-in INT8 activations on an
InstaCloud Intel Xeon 6975P-C. No VM restart or cache eviction occurs between
requests. Responses match byte-for-byte across paths and repetitions.

| Path | Measured request seconds | Median |
| --- | --- | ---: |
| Shared direct pipeline | 19.515750, 19.547507, 19.460840 | 19.515750 s |
| Resident HTTP endpoint | 19.238235, 19.114012, 19.255878 | 19.238235 s |

The approximately 1.4% difference is a small-sample observation on shared CPUs,
not proof that HTTP is faster. Both do one ASR window and one diarization pass.
The public comparison uses the HTTP median once; there is no separate, more
expensive HTTP inference algorithm. See [records](../benchmarks/results/single-pass/README.md).

## InstaCloud estimate

[Published rates](https://instacloud.com/pricing):

- CPU: $0.0004632 per vCPU-minute consumed.
- RAM: $0.0002316 per GB-minute consumed.
- Volume storage: $0.00000347 per GB-minute.
- Object storage: $0.000000347 per GB-minute.
- Egress: $0.05 per GB.

The README uses an explicit common scenario of **eight fully utilized vCPUs and
1.1 decimal GB RAM** throughout active processing. The RAM amount is a scenario
assumption, not a proven memory cap or a reconstruction of integrated usage.
Wall time alone does not reveal actual CPU utilization. These estimates are not
provider-metered charges and do not establish price superiority at equal quality.

```text
active_request_cost = 19.238235 / 60
                    * (8 * 0.0004632 + 1.1 * 0.0002316)
                    = $0.00126983894
cost_per_audio_hour = active_request_cost * 3600 / 27.27
                    = $0.16763550355
```

The calculation extrapolates many short requests, not one hour-long upload.
The direct pipeline median under the same assumptions gives approximately
$0.1701/audio-hour: effectively the same pipeline cost within this experiment's
variation, not a second product tier.

**Always-on idle residency is not free.** RAM remains billable between requests,
even with no audio submitted. The table estimates active processing only; total
cost per audio-hour depends on how much audio the service processes over its billed
uptime. Add idle CPU/RAM consumption, storage, egress, account fees, applicable
credits and taxes for a deployment budget. No startup or warm-up duration is
amortized into the displayed active-request estimate.

## Hosted API prices

The OpenAI comparator is
[`gpt-4o-transcribe-diarize`](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize),
and its model card lists $2.50 per million input tokens and $10 per million output
tokens. The [official pricing table](https://developers.openai.com/api/docs/pricing)
also supplies an **estimated $0.006/minute** for transcription plus diarization
(expand “All models” under transcription models). That estimate gives
**$0.36/audio-hour** and **$0.002727 per 27.27 seconds**. These are the provider's
duration-based estimates, not a flat-rate guarantee or an observed request charge;
no token counts or invoices were measured. It is a different model, not a renamed
Whisper Turbo endpoint. `whisper-1` is excluded from the combined-cost table
because it does not provide diarization.

[AssemblyAI's pre-recorded pricing](https://www.assemblyai.com/pricing) lists
Universal-2 at $0.15/audio-hour and Universal-3.5 Pro at $0.21/audio-hour, with
speaker diarization adding $0.02/audio-hour to either. Totals are $0.17 and $0.23;
27.27-second equivalents are $0.00128775 and $0.00174225. These are pre-recorded,
not streaming or Sync API prices. Free credits, custom discounts and other
optional add-ons are excluded. API latency, accuracy and actual billed amounts
were not measured, and no private recordings were submitted to either vendor.
