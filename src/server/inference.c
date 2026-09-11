#include "inference.h"
#include "alignment.h"
#include "languages.h"
#include "search.h"
#include "whisper_turbo_frontend.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
enum { WINDOW = 480000, EOT = 50257, SOT = 50258, TASK = 50360,
       NO_SPEECH = 50363, NO_TIMESTAMPS = 50364 };
static int language_token(uint32_t token, const char *language) {
    if (token < 50259 || token >= 50359)
        return 0;
    return !language || !strcmp(language, wt_languages[token - 50259]);
}
static void suppression(const cllm_whisper_turbo_decoder_weights *d, unsigned char *mask,
                        int first) {
    memset(mask, 0, CLLM_WHISPER_TURBO_VOCABULARY);
    for (size_t i = 0; i < d->suppress_count; ++i)
        if (d->suppress_ids[i] < CLLM_WHISPER_TURBO_VOCABULARY)
            mask[d->suppress_ids[i]] = 1;
    for (size_t i = EOT + 1; i < CLLM_WHISPER_TURBO_VOCABULARY; ++i)
        if (d->token_special[i] || i >= NO_TIMESTAMPS)
            mask[i] = 1;
    if (first) {
        mask[220] = 1;
        mask[EOT] = 1;
    }
}
typedef struct {
    const cllm_whisper_turbo_decoder_weights *weights;
    cllm_whisper_turbo_decoder_state *state;
    cllm_whisper_turbo_decoder_metrics *metrics;
    unsigned char *mask;
    float *cache[2];
    size_t bank, positions;
    wt_cancel cancel;
    void *cancel_context;
} beam_context;
enum { SELF_FLOATS = 4 * 448 * 1280, BEAM_FLOATS = 2 * SELF_FLOATS };

static void cache_copy(float *destination, const float *source, size_t positions) {
    for (size_t layer = 0; layer < 8; ++layer)
        memcpy(destination + layer * 448 * 1280, source + layer * 448 * 1280,
               positions * 1280 * sizeof(float));
}
static int beam_step(void *opaque, size_t beam, uint32_t token, int first, float *logits) {
    beam_context *c = opaque;
    if (c->cancel && c->cancel(c->cancel_context)) return -1;
    float *slot = c->cache[c->bank] + beam * BEAM_FLOATS;
    cache_copy(c->state->self_key, slot, c->positions);
    c->state->token_count = c->positions;
    c->state->logits = logits;
    suppression(c->weights, c->mask, first);
    uint32_t next; float score;
    int rc = cllm_whisper_turbo_decoder_step_filtered(c->weights, c->state, token,
                         c->mask, &next, &score, NULL, c->metrics);
    c->state->logits = NULL;
    if (rc || !isfinite(score)) return -1;
    cache_copy(slot, c->state->self_key, c->state->token_count);
    return 0;
}
static int beam_select(void *opaque, const size_t *parents, size_t count) {
    beam_context *c = opaque;
    ++c->positions;
    for (size_t i = 0; i < count; ++i)
        cache_copy(c->cache[1 - c->bank] + i * BEAM_FLOATS,
                   c->cache[c->bank] + parents[i] * BEAM_FLOATS, c->positions);
    c->bank = 1 - c->bank;
    return 0;
}
static int beam_fallback(const cllm_whisper_turbo_decoder_weights *d,
                          cllm_whisper_turbo_decoder_state *state, uint32_t language,
                          unsigned char *mask, cllm_whisper_turbo_decoder_metrics *metrics,
                          wt_cancel cancel, void *cancel_context,
                          uint32_t *tokens, size_t *count) {
    /* Cross K/V and the INT8 encoder result are reused. Only decoder hypotheses
       are retried; no audio splitting, forced language or partial completion. */
    float *alignment = state->alignment;
    state->alignment = NULL;
    state->token_count = 0;
    beam_context c = {.weights=d, .state=state, .metrics=metrics, .mask=mask,
                      .positions=3, .cancel=cancel, .cancel_context=cancel_context};
    int rc = -1;
    if (cllm_whisper_turbo_decoder_consume(d, state, SOT, NULL, metrics) ||
        cllm_whisper_turbo_decoder_consume(d, state, language, NULL, metrics) ||
        cllm_whisper_turbo_decoder_consume(d, state, TASK, NULL, metrics)) goto done;
    /* Sparse prefix copies only touch the used rows of these bounded banks. */
    for (size_t i = 0; i < 2; ++i) {
        c.cache[i] = calloc(WT_SEARCH_BEAMS * BEAM_FLOATS, sizeof(float));
        if (!c.cache[i]) goto done;
    }
    cache_copy(c.cache[0], state->self_key, 3);
    rc = wt_beam_search(CLLM_WHISPER_TURBO_VOCABULARY, EOT, NO_TIMESTAMPS,
                        WT_SEARCH_MAX_TEXT, beam_step, beam_select, &c, tokens, count);
done:
    free(c.cache[0]); free(c.cache[1]);
    state->logits = NULL;
    state->alignment = alignment;
    return rc;
}
static int sampling_fallback(const cllm_whisper_turbo_decoder_weights *d,
                              cllm_whisper_turbo_decoder_state *state, uint32_t language,
                              unsigned char *mask, float *logits,
                              cllm_whisper_turbo_decoder_metrics *metrics,
                              wt_cancel cancel, void *cancel_context,
                              uint32_t *output, size_t *count) {
    float *alignment = state->alignment;
    state->alignment = NULL;
    int rc = -1;
    for (unsigned stage = 1; stage <= 5; ++stage) {
        double temperature = stage / 5.0, best = -INFINITY;
        size_t best_count = 0;
        uint32_t best_tokens[WT_SEARCH_MAX_TEXT];
        for (unsigned attempt = 0; attempt < 5; ++attempt) {
            if (cancel && cancel(cancel_context)) goto done;
            state->token_count = 0;
            if (cllm_whisper_turbo_decoder_consume(d, state, SOT, NULL, metrics) ||
                cllm_whisper_turbo_decoder_consume(d, state, language, NULL, metrics) ||
                cllm_whisper_turbo_decoder_consume(d, state, TASK, NULL, metrics)) goto done;
            uint64_t rng = UINT64_C(0x243f6a8885a308d3) ^ ((uint64_t)stage << 32) ^ attempt;
            uint32_t tokens[WT_SEARCH_MAX_TEXT], token = NO_TIMESTAMPS;
            size_t n = 0;
            double sum_logprob = 0;
            int ended = 0;
            for (; n <= WT_SEARCH_MAX_TEXT; ++n) {
                if (cancel && cancel(cancel_context)) goto done;
                suppression(d, mask, n == 0);
                state->logits = logits;
                uint32_t next; float score;
                int step_rc = cllm_whisper_turbo_decoder_step_filtered(d, state, token,
                                      mask, &next, &score, NULL, metrics);
                state->logits = NULL;
                double logprob;
                if (step_rc || wt_sample_token(logits, CLLM_WHISPER_TURBO_VOCABULARY,
                                               temperature, &rng, &next, &logprob)) goto done;
                sum_logprob += logprob;
                if (next == EOT) { ended = 1; break; }
                if (n == WT_SEARCH_MAX_TEXT) break;
                tokens[n] = token = next;
            }
            int quality = ended ? wt_decode_quality(tokens, n, EOT, sum_logprob, 0) : WT_DECODE_RETRY;
            if (getenv("WHISPER_DIAGNOSTICS"))
                fprintf(stderr, "decoder: temperature=%.1f candidate=%u tokens=%zu ended=%d entropy=%.3f avg_logprob=%.3f accepted=%d\n",
                        temperature, attempt + 1, n, ended, wt_token_entropy(tokens, n, EOT),
                        sum_logprob / (n + 1), quality == WT_DECODE_ACCEPT);
            double average = sum_logprob / (n + 1);
            if (quality == WT_DECODE_ACCEPT && average > best) {
                best = average; best_count = n;
                memcpy(best_tokens, tokens, n * sizeof(*tokens));
            }
        }
        if (best_count) {
            memcpy(output, best_tokens, best_count * sizeof(*output)); *count = best_count;
            rc = 0; break;
        }
    }
done:
    state->logits = NULL;
    state->alignment = alignment;
    return rc;
}
static int transcribe(wt_engine *engine, const unsigned char *pcm, size_t samples,
                      const char *requested_language, wt_result *out, wt_error *error,
                      wt_cancel cancel, void *cancel_context, int aligned, const diar_result *speech) {
    if (!pcm || !samples || samples > WT_AUDIO_LIMIT)
        return wt_fail(error, 413, "Audio exceeds the supported sample limit.", "file",
                       "audio_too_long");
    const cllm_whisper_turbo_model *m = &engine->model;
    const cllm_whisper_turbo_decoder_weights *d = &m->decoder;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads((int)engine->threads);
#endif
    uint32_t language = 0;
    if (requested_language[0]) {
        for (uint32_t i = 50259; i < 50359; ++i)
            if (language_token(i, requested_language)) {
                language = i;
                break;
            }
        if (!language)
            return wt_fail(error, 400, "Language is not supported by the loaded model.", "language",
                           "invalid_value");
    }
    const size_t frames = cllm_whisper_turbo_log_mel_frames(WINDOW);
    const size_t encoded_frames = cllm_whisper_turbo_stem_output_frames(frames);
    const size_t scratch_count = cllm_whisper_turbo_log_mel_workspace_floats(WINDOW);
    float *audio = calloc(WINDOW, sizeof(float));
    float *mel = malloc(128 * frames * sizeof(float));
    float *scratch = malloc(scratch_count * sizeof(float));
    float *encoder = malloc(encoded_frames * 1280 * sizeof(float));
    unsigned char *mask = malloc(CLLM_WHISPER_TURBO_VOCABULARY);
    float *logits = malloc(CLLM_WHISPER_TURBO_VOCABULARY * sizeof(float));
    unsigned char *text = malloc(WT_TEXT_LIMIT + 1);
    float *alignment = aligned ? malloc(6 * 448 * encoded_frames * sizeof(float)) : NULL;
    const size_t word_capacity = ((samples + WINDOW - 1) / WINDOW) * 448;
    wt_word *words = aligned ? calloc(word_capacity, sizeof(wt_word)) : NULL;
    size_t word_count = 0;
    size_t length = 0;
    int result = -1;
    cllm_whisper_turbo_decoder_state state = {0};
    cllm_whisper_turbo_decoder_metrics metrics;
    cllm_whisper_turbo_encoder_metrics encoder_metrics;
    if (!audio || !mel || !scratch || !encoder || !mask || !logits || !text || (aligned && (!alignment || !words))) {
        wt_fail(error, 503, "Inference allocation failed.", NULL, "resource_exhausted");
        goto done;
    }
    for (size_t offset = 0; offset < samples; offset += WINDOW) {
        if (cancel && cancel(cancel_context))
            goto cancelled;
        size_t used = samples - offset < WINDOW ? samples - offset : WINDOW;
        /* Whole-input diarization supplies acoustic speech activity. Keep the
           original 30 s windows and original timestamps; skip only windows
           with no speech evidence (250 ms padding protects boundary speech).
           This is not per-speaker crop transcription or a text blacklist. */
        if (speech) {
            if (!wt_speech_window_active(speech, offset, used)) {
                if (getenv("WHISPER_DIAGNOSTICS"))
                    fprintf(stderr, "ASR window=%zu/%zu skipped: no acoustic speech activity\n",
                            offset / WINDOW + 1, (samples + WINDOW - 1) / WINDOW);
                continue;
            }
        }
        memset(audio, 0, WINDOW * sizeof(float));
        unsigned nonzero = 0;
        for (size_t i = 0; i < used; ++i) {
            const unsigned char *p = pcm + 2 * (offset + i);
            uint16_t value = (uint16_t)p[0] | (uint16_t)p[1] << 8;
            audio[i] = (int16_t)value / 32768.0f;
            nonzero |= value;
        }
        if (!nonzero)
            continue; /* Exact digital silence; no energy-threshold speech gating. */
        ++out->asr_windows;
        if (getenv("WHISPER_DIAGNOSTICS"))
            fprintf(stderr, "ASR window=%zu/%zu\n", offset / WINDOW + 1,
                    (samples + WINDOW - 1) / WINDOW);
        if (cllm_whisper_turbo_log_mel(audio, WINDOW, m->mel_filters, mel, scratch,
                                       scratch_count) ||
            cllm_whisper_turbo_encode_mel_cancel(m, mel, frames, 32, encoder, &encoder_metrics,
                                                 cancel, cancel_context))
            goto failed;
        if (cancel && cancel(cancel_context))
            goto cancelled;
        if (cllm_whisper_turbo_decoder_state_init(d, encoder, encoded_frames, 448, &state,
                                                  &metrics))
            goto failed;
        state.alignment = alignment;
        /* The no-speech probability must be measured before logit filtering. */
        state.logits = logits;
        uint32_t sot_next; float sot_score; double no_speech_logprob;
        int sot_rc = cllm_whisper_turbo_decoder_step(d, &state, SOT, &sot_next,
                                                     &sot_score, NULL, &metrics);
        state.logits = NULL;
        if (sot_rc || wt_logits_logprob(logits, CLLM_WHISPER_TURBO_VOCABULARY,
                                        NO_SPEECH, &no_speech_logprob)) goto failed;
        double no_speech_probability = exp(no_speech_logprob);
        if (!language) {
            float score = -INFINITY;
            for (uint32_t i = 50259; i < 50359; ++i)
                if (language_token(i, NULL) && logits[i] > score) { language = i; score = logits[i]; }
            if (!language_token(language, NULL) || !isfinite(score)) goto failed;
        }
        if (cllm_whisper_turbo_decoder_consume(d, &state, language, NULL, &metrics) ||
            cllm_whisper_turbo_decoder_consume(d, &state, TASK, NULL, &metrics))
            goto failed;
        uint32_t token = NO_TIMESTAMPS, next = 0;
        int ended = 0;
        size_t before = length;
        size_t token_offsets[449], token_count = 0;
        uint32_t generated_tokens[444];
        double sum_logprob = 0;
        for (unsigned generated = 0; generated + 4 < 448; ++generated) {
            if (cancel && cancel(cancel_context))
                goto cancelled;
            suppression(d, mask, generated == 0);
            float score;
            state.logits = logits;
            int step_rc = cllm_whisper_turbo_decoder_step_filtered(d, &state, token, mask,
                                                                    &next, &score, NULL, &metrics);
            state.logits = NULL;
            double logprob;
            if (step_rc || !isfinite(score) || wt_logits_logprob(logits,
                CLLM_WHISPER_TURBO_VOCABULARY, next, &logprob) || !isfinite(logprob)) goto failed;
            sum_logprob += logprob;
            if (next == EOT) {
                ended = 1;
                break;
            }
            if (next >= EOT || d->token_special[next])
                goto failed;
            generated_tokens[token_count] = next;
            token_offsets[token_count++] = length;
            size_t start = d->token_offsets[next], piece = d->token_offsets[next + 1] - start;
            if (piece > WT_TEXT_LIMIT - length) {
                wt_fail(error, 422, "Transcription exceeds the output limit.", "file",
                        "output_limit_exceeded");
                goto done;
            }
            memcpy(text + length, d->token_bytes + start, piece);
            length += piece;
            token = next;
        }
        int quality = ended ? wt_decode_quality(generated_tokens, token_count, EOT,
                                                sum_logprob, no_speech_probability) : WT_DECODE_RETRY;
        if (getenv("WHISPER_DIAGNOSTICS"))
            fprintf(stderr, "decoder: greedy tokens=%zu ended=%d entropy=%.3f avg_logprob=%.3f no_speech=%.3f quality=%d\n",
                    token_count, ended, wt_token_entropy(generated_tokens, token_count, EOT),
                    sum_logprob / (token_count + 1), no_speech_probability, quality);
        if (quality == WT_DECODE_SILENCE) {
            length = before;
            cllm_whisper_turbo_decoder_state_free(&state);
            continue;
        }
        if (quality == WT_DECODE_RETRY) {
            uint32_t recovered[WT_SEARCH_MAX_TEXT];
            size_t recovered_count = 0;
            if (getenv("WHISPER_DIAGNOSTICS"))
                fprintf(stderr, "decoder: greedy rejected; retry with %d quality-gated beams\n", WT_SEARCH_BEAMS);
            if (beam_fallback(d, &state, language, mask, &metrics, cancel, cancel_context,
                               recovered, &recovered_count) &&
                sampling_fallback(d, &state, language, mask, logits, &metrics,
                                   cancel, cancel_context, recovered, &recovered_count)) {
                if (cancel && cancel(cancel_context)) goto cancelled;
                wt_fail(error, 422, "No complete transcription passed decoder quality checks within the retry budget.",
                        "file", "decoding_failed");
                goto done;
            }
            length = before;
            token_count = recovered_count;
            /* Capture alignment for the winning hypothesis only, reusing the
               cross caches. No encoder or diarization pass is repeated. */
            if (aligned) {
                state.token_count = 0;
                uint32_t prefix[] = {SOT, language, TASK, NO_TIMESTAMPS};
                for (size_t p = 0; p < 4; ++p)
                    if (cllm_whisper_turbo_decoder_consume(d, &state, prefix[p], NULL, &metrics)) goto failed;
            }
            for (size_t t = 0; t < token_count; ++t) {
                if (cancel && cancel(cancel_context)) goto cancelled;
                uint32_t id = recovered[t];
                if (id >= EOT || d->token_special[id]) goto failed;
                token_offsets[t] = length;
                size_t start = d->token_offsets[id], piece = d->token_offsets[id + 1] - start;
                if (piece > WT_TEXT_LIMIT - length) {
                    wt_fail(error, 422, "Transcription exceeds the output limit.", "file", "output_limit_exceeded");
                    goto done;
                }
                memcpy(text + length, d->token_bytes + start, piece); length += piece;
                if (aligned && cllm_whisper_turbo_decoder_consume(d, &state, id, NULL, &metrics)) goto failed;
            }
            if (getenv("WHISPER_DIAGNOSTICS"))
                fprintf(stderr, "decoder: recovered %zu text tokens with quality checks and model-selected EOT\n", token_count);
        }
        token_offsets[token_count] = length;
        if (aligned && token_count) {
            /* The last generated text token has already been consumed when EOT
               is predicted. Consume EOT only for the normalization sentinel;
               reuse all encoder/self/cross caches, without another ASR pass. */
            if (cllm_whisper_turbo_decoder_consume(d, &state, EOT, NULL, &metrics)) goto failed;
            size_t bounds[449], real_frames = used / 320;
            if (!real_frames) real_frames = 1;
            if (wt_align(alignment, state.token_count, token_count, real_frames,
                         encoded_frames, 448, bounds)) goto failed;
            if (wt_alignment_words(text, length, token_offsets, bounds, token_count,
                                    offset, samples, words, word_capacity, &word_count)) {
                wt_fail(error, 422, "Cannot obtain a positive-duration text alignment.",
                        "file", "alignment_failed");
                goto done;
            }
        }
        cllm_whisper_turbo_decoder_state_free(&state);
        if (length > before && offset + used < samples && text[length - 1] != ' ') {
            if (length == WT_TEXT_LIMIT)
                goto failed;
            text[length++] = ' ';
            if (aligned && word_count) ++words[word_count - 1].length;
        }
    }
    /* Match the text endpoint's surrounding-whitespace behavior without changing
     * words. */
    while (length && (text[length - 1] == ' ' || text[length - 1] == '\n'))
        --length;
    size_t first = 0;
    while (first < length && (text[first] == ' ' || text[first] == '\n'))
        ++first;
    memmove(text, text + first, length - first);
    length -= first;
    for (size_t i = 0; i < word_count; ++i) {
        size_t a = words[i].offset, b = a + words[i].length;
        if (a < first) a = first;
        if (b > first + length) b = first + length;
        words[i].offset = a - first;
        words[i].length = b > a ? b - a : 0;
    }
    text[length] = 0;
    out->text = text;
    out->length = length;
    out->duration = samples / 16000.0;
    out->words = words;
    out->word_count = word_count;
    words = NULL;
    text = NULL;
    result = 0;
    goto done;
failed:
    if (cancel && cancel(cancel_context))
        goto cancelled;
    wt_fail(error, 500, "Inference failed.", NULL, "inference_error");
    goto done;
cancelled:
    wt_fail(error, 504, "Transcription cancelled or deadline exceeded.", NULL, "request_timeout");
done:
    cllm_whisper_turbo_decoder_state_free(&state);
    free(audio);
    free(mel);
    free(scratch);
    free(encoder);
    free(mask);
    free(logits);
    free(text);
    free(alignment);
    free(words);
    return result;
}
int wt_transcribe_pcm(wt_engine *engine, const unsigned char *pcm, size_t samples,
                       const char *language, wt_result *out, wt_error *error,
                       wt_cancel cancel, void *context) {
    return transcribe(engine, pcm, samples, language, out, error, cancel, context, 0, NULL);
}
int wt_transcribe_aligned_pcm(wt_engine *engine, const unsigned char *pcm, size_t samples,
                               const char *language, wt_result *out, wt_error *error,
                               wt_cancel cancel, void *context) {
    return transcribe(engine, pcm, samples, language, out, error, cancel, context, 1, NULL);
}
int wt_transcribe_speech_pcm(wt_engine *engine, const unsigned char *pcm, size_t samples,
                            const char *language, wt_result *out, wt_error *error,
                            wt_cancel cancel, void *context, const diar_result *speech) {
    return transcribe(engine, pcm, samples, language, out, error, cancel, context, 1, speech);
}
