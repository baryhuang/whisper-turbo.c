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
recording, English decoding, four threads and INT8 weights and encoder activations on an
InstaCloud Intel Xeon 6975P-C configured for four vCPUs. Both processes are pinned
to four logical CPUs. No VM restart or cache eviction occurs between
requests. Responses match byte-for-byte across paths and repetitions.

| Path | Measured request seconds | Median |
| --- | --- | ---: |
| Resident direct CLI | 35.308962, 35.237685, 35.273096 | 35.273096 s |
| Resident HTTP endpoint | 36.298806, 35.537225, 35.739450 | 35.739450 s |

Both paths do one ASR window and one diarization pass. The cost comparison uses
the HTTP median. See [records](../benchmarks/results/single-pass/README.md).

## InstaCloud estimate

[Published rates](https://instacloud.com/pricing):

- CPU: $0.0004632 per vCPU-minute consumed.
- RAM: $0.0002316 per GB-minute consumed.
- Volume storage: $0.00000347 per GB-minute.
- Object storage: $0.000000347 per GB-minute.
- Egress: $0.05 per GB.

The README uses an explicit common scenario of **four fully utilized vCPUs and
1.1 decimal GB RAM** throughout active processing. The RAM amount is a scenario
assumption, not a proven memory cap or a reconstruction of integrated usage.
Wall time alone does not reveal actual CPU utilization. These estimates are not
provider-metered charges and do not establish price superiority at equal quality.

```text
active_request_cost = 35.739450 / 60
                    * (4 * 0.0004632 + 1.1 * 0.0002316)
                    = $0.00125538392
cost_per_audio_hour = active_request_cost * 3600 / 27.27
                    = $0.16572725026
```

The calculation extrapolates many short requests, not one hour-long upload.

**Always-on idle residency is not free.** RAM remains billable between requests,
even with no audio submitted. The table estimates active processing only; total
cost per audio-hour depends on how much audio the service processes over its billed
uptime. Add idle CPU/RAM consumption, storage, egress, account fees, applicable
credits and taxes for a deployment budget. No startup or warm-up duration is
amortized into the displayed active-request estimate.

## CPU scaling

The same HTTP workload on eight CPUs took 19.238235 seconds (median), compared
with 35.739450 seconds on four CPUs: a 1.8577× speedup. The active-request cost
formula above, using eight CPUs and the same 1.1 GB RAM assumption, gives
$0.00126983894 per clip and $0.16763550355 per audio hour.

Four CPUs use about 143 CPU-seconds at full utilization, versus 154 for eight
CPUs, but keep memory occupied longer. That is why halving CPU count does not
halve estimated cost. Both runs use INT8 weights and encoder activations, the
same binaries, CPU model and input. Results come from separate runs on shared
CPUs; they are not metered invoices or a general scaling guarantee.

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
