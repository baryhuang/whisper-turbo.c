#include "inference.h"
#include "alignment.h"
#include "network.h"
#include "pipeline.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static int reference_names(const wt_engine *engine, const wt_request *r, const diar_result *d,
                           char names[32][64], wt_error *e, wt_cancel cancel, void *context) {
    for (int s = 0; s < 32; ++s)
        names[s][0] = 0;
    if (!r->name_count)
        return 0;
    diar_checkpoint seg = {0}, emb = {0};
    char path[4096];
    int rc = -1;
    float *audio = calloc(DIAR_SAMPLES, sizeof(float));
    if (!audio)
        return wt_fail(e, 503, "Reference allocation failed.", NULL, "resource_exhausted");
    if (snprintf(path, sizeof(path), "%s/segmentation-pytorch_model.bin",
                 engine->diarization_directory) >= (int)sizeof(path) ||
        diar_checkpoint_open(&seg, path))
        goto failed;
    if (snprintf(path, sizeof(path), "%s/embedding-pytorch_model.bin",
                 engine->diarization_directory) >= (int)sizeof(path) ||
        diar_checkpoint_open(&emb, path))
        goto failed;
    double scores[4][32];
    for (unsigned k = 0; k < r->name_count; ++k) {
        if (cancel && cancel(context)) {
            wt_fail(e, 504, "Reference processing cancelled.", NULL, "request_timeout");
            goto done;
        }
        unsigned char *pcm = NULL;
        size_t samples = 0;
        if (wt_reference_wav(r->references[k], r->reference_lengths[k], &pcm, &samples, e))
            goto done;
        memset(audio, 0, DIAR_SAMPLES * sizeof(float));
        for (size_t i = 0; i < samples; ++i)
            audio[i] = (int16_t)(pcm[i * 2] | (unsigned)pcm[i * 2 + 1] << 8) / 32768.0f;
        free(pcm);
        float p[DIAR_FRAMES * 7], mask[DIAR_FRAMES * 3], v[3 * 256];
        if (diar_segment(&seg, audio, p))
            goto failed;
        diar_powerset(p, mask, DIAR_FRAMES);
        int counts[3] = {0};
        for (int t = 0; t < DIAR_FRAMES; ++t) {
            int active = 0;
            for (int s = 0; s < 3; ++s)
                active += mask[t * 3 + s] > 0;
            for (int s = 0; s < 3; ++s) {
                if (t * 270 + 495.5 >= samples || active > 1)
                    mask[t * 3 + s] = 0;
                counts[s] += mask[t * 3 + s] > 0;
            }
        }
        int dominant = counts[1] > counts[0] ? 1 : 0;
        if (counts[2] > counts[dominant])
            dominant = 2;
        if (counts[dominant] < 20) {
            wt_fail(e, 400, "Speaker reference has insufficient clean speech.",
                    "known_speaker_references", "invalid_audio");
            goto done;
        }
        if (cancel && cancel(context)) {
            wt_fail(e, 504, "Reference processing cancelled.", NULL, "request_timeout");
            goto done;
        }
        if (diar_embed(&emb, audio, mask, v))
            goto failed;
        for (int s = 0; s < d->speakers; ++s) {
            double dot = 0, a = 0, b = 0;
            for (int j = 0; j < 256; ++j) {
                double x = v[dominant * 256 + j], y = d->centroids[s * 256 + j];
                dot += x * y;
                a += x * x;
                b += y * y;
            }
            scores[k][s] = a > 0 && b > 0 ? dot / sqrt(a * b) : -1;
        }
    }
    /* Conservative one-to-one cosine matching; ambiguous or weak references
       leave the cluster anonymous. This threshold is not identity certification.
     */
    unsigned used = 0;
    for (unsigned pass = 0; pass < r->name_count; ++pass) {
        double best = 0.65;
        int bk = -1, bs = -1;
        for (unsigned k = 0; k < r->name_count; ++k)
            if (!(used & (1U << k))) {
                double first = -1, second = -1;
                int speaker = -1;
                for (int s = 0; s < d->speakers; ++s) {
                    double v = scores[k][s];
                    if (v > first) {
                        second = first;
                        first = v;
                        speaker = s;
                    } else if (v > second)
                        second = v;
                }
                if (speaker >= 0 && !names[speaker][0] && first > best && first - second >= 0.05) {
                    best = first;
                    bk = (int)k;
                    bs = speaker;
                }
            }
        if (bk < 0)
            break;
        memcpy(names[bs], r->names[bk], sizeof(names[bs]));
        used |= 1U << bk;
    }
    rc = 0;
    goto done;
failed:
    wt_fail(e, 500, "Speaker reference inference failed.", NULL, "diarization_error");
done:
    diar_checkpoint_close(&seg);
    diar_checkpoint_close(&emb);
    free(audio);
    return rc;
}
int wt_transcribe(void *opaque, const wt_request *r, wt_result *out, wt_error *e, wt_cancel cancel,
                  void *context) {
    wt_engine *engine = opaque;
    const unsigned char *pcm;
    size_t samples;
    if (wt_wav(r->file, r->file_size, &pcm, &samples, e))
        return -1;
    if (!r->diarize)
        return wt_transcribe_pcm(engine, pcm, samples, r->language, out, e, cancel, context);
    if (!engine->diarization_directory)
        return wt_fail(e, 503, "Configure WHISPER_DIARIZATION_MODELS to enable diarization.",
                       "model", "model_unavailable");
    if (samples > 480000 && !r->chunk_auto)
        return wt_fail(e, 400, "Audio over 30 seconds requires chunking_strategy=auto.",
                       "chunking_strategy", "missing_required_parameter");
    /* Validate every untrusted reference before expensive inference. */
    for (unsigned i = 0; i < r->reference_count; ++i) {
        unsigned char *ref = NULL;
        size_t n = 0;
        if (wt_reference_wav(r->references[i], r->reference_lengths[i], &ref, &n, e))
            return -1;
        free(ref);
    }
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads((int)engine->threads);
#endif
    /* One full-recording diarization pass supplies speech activity, followed by
       one full-recording ASR pass and alignment-based speaker assignment.
       Workspaces stay sequential; never decode individual speaker turns. */
    diar_result d = {0};
    float *audio = malloc(samples * sizeof(float));
    if (!audio) {
        wt_fail(e, 503, "Audio allocation failed.", NULL, "resource_exhausted");
        goto failed;
    }
    unsigned nonzero = 0;
    for (size_t i = 0; i < samples; ++i) {
        unsigned v = pcm[i * 2] | (unsigned)pcm[i * 2 + 1] << 8;
        audio[i] = (int16_t)v / 32768.0f;
        nonzero |= v;
    }
    int rc = 0;
    if (nonzero) {
        ++out->diarization_passes;
        rc = diar_run(engine->diarization_directory, audio, samples, 0, &d, cancel, context);
    }
    free(audio);
    if (rc) {
        if (cancel && cancel(context))
            wt_fail(e, 504, "Diarization cancelled.", NULL, "request_timeout");
        else
            wt_fail(e, 500, "Diarization failed; no speaker labels fabricated.", NULL, "diarization_error");
        goto failed;
    }
    if (wt_transcribe_speech_pcm(engine, pcm, samples, r->language, out, e, cancel, context, &d))
        goto failed;
    char names[32][64];
    if (reference_names(engine, r, &d, names, e, cancel, context) ||
        wt_assign_speakers(out, &d, (const char (*)[64])names, r, e))
        goto failed;
    diar_result_free(&d);
    return 0;
failed:
    diar_result_free(&d);
    wt_result_free(out);
    return -1;
}
