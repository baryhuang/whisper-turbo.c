# ASR and diarization cost methodology

Prices checked September 11, 2026; USD. The native comparison uses an always-on,
already-warmed service. **Startup, model initialization, warm-up and shutdown are
excluded from both reported latency and the active-request cost calculation.**

## Current optimized results

Intel Xeon 6975P-C, eight shared vCPUs, eight threads, one resident model.
Each engine runs sequentially on the same machine with one discarded warm-up.

| Workload | Actual audio | Baseline processing | Optimized processing | Baseline cost | Optimized cost | Estimated $/audio hour, baseline → optimized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASR, 10 LibriSpeech clips | 91.525 s total | 107.109957 s total | 92.604596 s total | $0.00706990 | $0.00611246 | $0.278084 → $0.240424 |
| ASR + alignment + diarization, synthetic A–B–A | 26 s / request | 18.798637 s mean | 14.8762855 s mean | $0.00124082 | $0.00098192 | $0.171806 → $0.135959 |

Combined measured requests were **19.100611, 18.496663 seconds** before and
**14.762573, 14.989998 seconds** after optimization. ASR totals sum ten requests,
not ten repetitions of the combined fixture. Time and scenario cost decrease
**13.5% for ASR** and **20.9% for combined processing**. Normalize each workload
by its own audio duration; do not compare ASR-only and combined cost as if their
inputs were the same.

ASR transcripts are identical (3 errors / 251 words); combined diarized JSON is
byte-identical. Both builds use INT8 weights and encoder activations; the optimized
build adds INT8 decoder projections, AVX-512 FP32 diarization and batched LSTM
input projections. See [configuration and limitations](cpu-optimizations.md) and
[raw measurements](../benchmarks/results/cpu-optimizations/measurements.json).
An interrupted exit-137 attempt completed no measured requests and is excluded;
it remains recorded in the raw results. These tests are not a production bill,
a broad accuracy comparison, or a measurement of an hour-long upload.

## InstaCloud estimate

[Published rates](https://instacloud.com/pricing), checked September 11, 2026:

| Resource | USD rate |
| --- | ---: |
| CPU | $0.0004632 / vCPU-minute |
| RAM | $0.0002316 / GB-minute |

Use **eight fully utilized vCPUs and 1.1 decimal GB RAM** throughout active
processing for both builds. The RAM assumption preserves the earlier cost model;
it is neither measured average residency nor the configured memory allocation.
Lifetime memory peaks cannot substitute for integrated billable RAM usage.
Wall time alone does not reveal actual CPU utilization.

```text
resource_cost_per_minute = 8 * 0.0004632 + 1.1 * 0.0002316
                         = $0.00396036
workload_cost = processing_seconds / 60 * resource_cost_per_minute
cost_per_audio_hour = workload_cost * 3600 / actual_audio_seconds

combined_before = 18.798637 / 60 * 0.00396036 = $0.001240822833822
combined_after = 14.8762855 / 60 * 0.00396036 = $0.000981924100713
combined_after_per_audio_hour = 0.000981924100713 * 3600 / 26
                              = $0.135958721637
```

All displayed results use unrounded input measurements; rounding occurs only
for presentation. Percentage reductions use the same resource assumptions for
both builds, not different CPU allocations or different clips.

**Active processing only, not metered production savings.** Startup, loading,
warm-up, idle residency, storage, network, account fees, credits and taxes are
excluded. An always-on service also consumes billable memory between requests;
total deployment cost per audio-hour depends on throughput over its billed uptime.
These estimates do not establish price superiority at equal quality.

## Historical resident-path comparison

The following 27.27-second, four-CPU measurements predate the new optimizations.
They remain separate from the current 26-second, eight-CPU fixture above.

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

Both paths do one ASR window and one diarization pass.
See [records](../benchmarks/results/single-pass/README.md).
The historical estimate uses the HTTP median, four fully utilized CPUs and the
same 1.1 GB RAM assumption:

```text
active_request_cost = 35.739450 / 60
                    * (4 * 0.0004632 + 1.1 * 0.0002316)
                    = $0.00125538392
cost_per_audio_hour = active_request_cost * 3600 / 27.27
                    = $0.16572725026
```

## Historical CPU scaling

The same HTTP workload on eight CPUs took 19.238235 seconds (median), compared
with 35.739450 seconds on four CPUs: a 1.8577× speedup. The active-request cost
formula above, using eight CPUs and the same 1.1 GB RAM assumption, gives
$0.00126983894 per clip and $0.16763550355 per audio hour.

Four CPUs use about 143 CPU-seconds at full utilization, versus 154 for eight
CPUs, but keep memory occupied longer. That is why halving CPU count does not
halve estimated cost. Both runs use INT8 weights and encoder activations, the
same binaries, CPU model and input. Results come from separate runs on shared
CPUs; they are not metered invoices or a general scaling guarantee.

## WhisperX public pricing comparison

The README uses **WhisperX + pyannote hosted by Whipscribe**, rather than a
self-hosted resource estimate. The provider
[identifies its backend as WhisperX and pyannote running on GPUs](https://whipscribe.com/blog/whisperx-vs-whipscribe-2026).
Its [published pricing](https://whipscribe.com/pricing), checked September 11, 2026,
includes speaker labels at no extra charge:

| Credit pack | Audio minutes | Effective $/audio hour |
| --- | ---: | ---: |
| $8 Starter | 1,000 | $0.480 |
| $12 | 2,000 | $0.360 |
| $24 Team, including API access | 5,000 | $0.288 |

The comparison selects the Team pack: **$24 / 5,000 × 60 = $0.288/audio-hour**.
This requires a $24 upfront purchase and full use of the pack to realize that
effective rate. Credits do not expire according to the current pricing page.
These are hosted-product prices, not WhisperX software fees, GPU rental rates,
or measurements of our CPU deployment. Displayed hourly comparisons use three
decimal places; underlying benchmark calculations retain full precision.

[WhisperX v3.8.6](https://github.com/m-bain/whisperX/tree/v3.8.6) supports CPU-only
execution; a GPU is optional. The documented CPU ASR settings are
`--device cpu --compute_type int8`. Forced alignment and pyannote speaker
diarization also accept the CPU device; the
[diarization wrapper](https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/diarize.py)
defaults to CPU. ASR INT8 does not quantize the separate alignment and diarization
models. See the [upstream CPU instructions](https://github.com/m-bain/whisperX/blob/v3.8.6/README.md#usage--command-line).

WhisperX's published transcription speed alone is not used to estimate the full
pipeline: alignment and diarization add work. The hosted price avoids conflating
GPU-hours with audio-hours or presenting GPU throughput as a CPU benchmark.

## Cloud API prices

The OpenAI comparator is
[`gpt-4o-transcribe-diarize`](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize),
and its model card lists $2.50 per million input tokens and $10 per million output
tokens. The [official pricing table](https://developers.openai.com/api/docs/pricing)
also supplies an **estimated $0.006/minute** for transcription plus diarization
(expand “All models” under transcription models). That estimate gives
**$0.36/audio-hour** and **$0.002600 per 26 seconds**. These are the provider's
duration-based estimates, not a flat-rate guarantee or an observed request charge;
no token counts or invoices were measured. It is a different model, not a renamed
Whisper Turbo endpoint. `whisper-1` is excluded from the combined-cost table
because it does not provide diarization.

[AssemblyAI's pre-recorded pricing](https://www.assemblyai.com/pricing) lists
Universal-2 at $0.15/audio-hour and Universal-3.5 Pro at $0.21/audio-hour, with
speaker diarization adding $0.02/audio-hour to either. Totals are $0.17 and $0.23;
26-second equivalents are $0.00122778 and $0.00166111. These are pre-recorded,
not streaming or Sync API prices. Free credits, custom discounts and other
optional add-ons are excluded. API latency, accuracy and actual billed amounts
were not measured, and no private recordings were submitted to either vendor.

### Azure, Google Cloud and AWS batch transcription

Published USD prices checked September 11, 2026, for transcription **with speaker
diarization**, sorted by cost per audio hour. These are Cloud API charges, not
VM rental estimates. Rates assume mono audio and exclude storage, network, taxes,
free credits, commitments and negotiated discounts.

| Service / tier | Published rate | Cost per audio hour |
| --- | ---: | ---: |
| Azure Speech Standard batch, East US | $0.180 / hour | $0.180 |
| Google Cloud Speech-to-Text V2 dynamic batch | $0.003 / minute | $0.180 |
| Amazon Transcribe Standard batch, US East (N. Virginia) | $0.006 / minute | $0.360 |

**Azure:** The [Azure Retail Prices API](https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27eastus%27%20and%20meterName%20eq%20%27S1%20Speech%20to%20Text%20Batch%27)
lists `S1 Speech to Text Batch` at $0.180 per hour in `eastus`, meter
`48ff31f2-a620-5227-8bc7-4c67b026040e`, with a zero-unit tier minimum.
The [Speech pricing page](https://azure.microsoft.com/en-us/pricing/details/speech/)
includes diarization in batch pricing at no extra charge; the batch rate requires
Speech-to-text REST API v3.2 or later. This is not the real-time or Fast
Transcription tier.

**Google Cloud:** [V2 pricing](https://cloud.google.com/speech-to-text/pricing)
lists Standard dynamic batch at $0.003 per minute: $0.003 × 60 = **$0.180/audio-hour**.
Use a diarization-capable model and supported language/region, such as
[Chirp 3 with English (US) in the US multi-region](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3).
Diarization is part of the recognition configuration, without a separate listed
add-on charge. The discounted
[`DYNAMIC_BATCHING` processing strategy](https://docs.cloud.google.com/speech-to-text/docs/reference/rest/v2/projects.locations.recognizers/batchRecognize)
allows up to 24 hours for completion. Ordinary V2 Standard recognition is
$0.016 per minute (**$0.960/audio-hour**) at the first usage tier; it is not the
discounted mode selected in the README.

**AWS:** [Amazon Transcribe pricing](https://aws.amazon.com/transcribe/pricing/)
lists Standard batch at $0.006 per minute in its US East (N. Virginia) example:
$0.006 × 60 = **$0.360/audio-hour**. Speaker diarization is included in Standard
pricing. This excludes Call Analytics, medical transcription, custom language
models and content-redaction add-ons.
