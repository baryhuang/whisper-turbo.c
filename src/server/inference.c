#include "inference.h"
#include "languages.h"
#include "whisper_turbo_frontend.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
enum { WINDOW = 480000, EOT = 50257, SOT = 50258, TASK = 50360, NO_TIMESTAMPS = 50364 };
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
int wt_transcribe(void *opaque, const wt_request *r, wt_result *out, wt_error *error,
                  wt_cancel cancel, void *cancel_context) {
    wt_engine *engine = opaque;
    const cllm_whisper_turbo_model *m = &engine->model;
    const cllm_whisper_turbo_decoder_weights *d = &m->decoder;
    const unsigned char *pcm;
    size_t samples;
    if (wt_wav(r->file, r->file_size, &pcm, &samples, error))
        return -1;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads((int)engine->threads);
#endif
    uint32_t language = 0;
    if (r->language[0]) {
        for (uint32_t i = 50259; i < 50359; ++i)
            if (language_token(i, r->language)) {
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
    unsigned char *text = malloc(WT_TEXT_LIMIT + 1);
    size_t length = 0;
    int result = -1;
    cllm_whisper_turbo_decoder_state state = {0};
    cllm_whisper_turbo_decoder_metrics metrics;
    cllm_whisper_turbo_encoder_metrics encoder_metrics;
    if (!audio || !mel || !scratch || !encoder || !mask || !text) {
        wt_fail(error, 503, "Inference allocation failed.", NULL, "resource_exhausted");
        goto done;
    }
    for (size_t offset = 0; offset < samples; offset += WINDOW) {
        if (cancel && cancel(cancel_context))
            goto cancelled;
        size_t used = samples - offset < WINDOW ? samples - offset : WINDOW;
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
        if (!language) {
            memset(mask, 1, CLLM_WHISPER_TURBO_VOCABULARY);
            for (uint32_t i = 50259; i < 50359; ++i)
                if (language_token(i, NULL))
                    mask[i] = 0;
            float score;
            if (cllm_whisper_turbo_decoder_step_filtered(d, &state, SOT, mask, &language, &score,
                                                         NULL, &metrics) ||
                !language_token(language, NULL) || !isfinite(score))
                goto failed;
        } else if (cllm_whisper_turbo_decoder_consume(d, &state, SOT, NULL, &metrics))
            goto failed;
        if (cllm_whisper_turbo_decoder_consume(d, &state, language, NULL, &metrics) ||
            cllm_whisper_turbo_decoder_consume(d, &state, TASK, NULL, &metrics))
            goto failed;
        uint32_t token = NO_TIMESTAMPS, next = 0;
        int ended = 0;
        size_t before = length;
        for (unsigned generated = 0; generated + 4 < 448; ++generated) {
            if (cancel && cancel(cancel_context))
                goto cancelled;
            suppression(d, mask, generated == 0);
            float score;
            if (cllm_whisper_turbo_decoder_step_filtered(d, &state, token, mask, &next, &score,
                                                         NULL, &metrics) ||
                !isfinite(score))
                goto failed;
            if (next == EOT) {
                ended = 1;
                break;
            }
            if (next >= EOT || d->token_special[next])
                goto failed;
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
        cllm_whisper_turbo_decoder_state_free(&state);
        if (!ended) {
            wt_fail(error, 422, "Decoder context exhausted; use shorter audio segments.", "file",
                    "context_length_exceeded");
            goto done;
        }
        if (length > before && offset + used < samples && text[length - 1] != ' ') {
            if (length == WT_TEXT_LIMIT)
                goto failed;
            text[length++] = ' ';
        }
    }
    /* Match the text endpoint's surrounding-whitespace behavior without changing words. */
    while (length && (text[length - 1] == ' ' || text[length - 1] == '\n'))
        --length;
    size_t first = 0;
    while (first < length && (text[first] == ' ' || text[first] == '\n'))
        ++first;
    memmove(text, text + first, length - first);
    length -= first;
    text[length] = 0;
    out->text = text;
    out->length = length;
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
    free(text);
    return result;
}
