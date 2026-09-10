#define _POSIX_C_SOURCE 200809L
#include "pipeline.h"
#include "cluster.h"
#include "network.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static int assign(const float *e, const float *mask, const float *cent, int k, int *out) {
    if (k < 1 || k > 32)
        return -1;
    double cost[3][32], min = INFINITY;
    int active[3] = {0};
    for (int s = 0; s < 3; ++s) {
        double norm = 0;
        for (int d = 0; d < 256; ++d)
            norm += (double)e[s * 256 + d] * e[s * 256 + d];
        for (int t = 0; t < DIAR_FRAMES; ++t)
            active[s] += mask[t * 3 + s] > 0;
        for (int c = 0; c < k; ++c) {
            double dot = 0, cnorm = 0;
            for (int d = 0; d < 256; ++d) {
                dot += (double)e[s * 256 + d] * cent[c * 256 + d];
                cnorm += (double)cent[c * 256 + d] * cent[c * 256 + d];
            }
            cost[s][c] = 1 + dot / sqrt(norm * cnorm);
            if (isfinite(cost[s][c]))
                min = fmin(min, cost[s][c]);
        }
    }
    if (!isfinite(min))
        return -1;
    for (int s = 0; s < 3; ++s)
        for (int c = 0; c < k; ++c) {
            if (!isfinite(cost[s][c]))
                cost[s][c] = min;
            if (!active[s])
                cost[s][c] = min - 1;
        }
    double best = -INFINITY;
    int begin = k < 3 ? -1 : 0;
    for (int a = begin; a < k; ++a)
        for (int b = begin; b < k; ++b)
            for (int c = begin; c < k; ++c) {
                int selected = (a >= 0) + (b >= 0) + (c >= 0);
                if (selected != (k < 3 ? k : 3) || (a >= 0 && a == b) || (a >= 0 && a == c) ||
                    (b >= 0 && b == c))
                    continue;
                double score = (a >= 0 ? cost[0][a] : 0) + (b >= 0 ? cost[1][b] : 0) +
                               (c >= 0 ? cost[2][c] : 0);
                if (score > best) {
                    best = score;
                    out[0] = a;
                    out[1] = b;
                    out[2] = c;
                }
            }
    for (int s = 0; s < 3; ++s)
        if (!active[s])
            out[s] = -1;
    return 0;
}
typedef diar_interval segment;
static int order(const void *a, const void *b) {
    const segment *x = a, *y = b;
    if (x->start != y->start)
        return x->start < y->start ? -1 : 1;
    return x->speaker - y->speaker;
}
static int intervals(const uint32_t *bits, size_t frames, int k, const int *labels, double duration,
                     diar_interval **out, size_t *count) {
    size_t cap = frames * 2 + 32, n = 0;
    segment *segs = calloc(cap, sizeof(segment));
    if (!segs)
        return -1;
    for (int s = 0; s < k; ++s) {
        size_t start = 0;
        int on = 0;
        for (size_t t = 0; t <= frames; ++t) {
            int active = t < frames && ((bits[t] >> s) & 1);
            if (active && !on) {
                start = t;
                on = 1;
            }
            if (!active && on) {
                double a = (start * 270 + 495.5) / 16000.0,
                       b = fmin(duration, (t * 270 + 495.5) / 16000.0);
                if (b > a) {
                    if (n == cap) {
                        free(segs);
                        return -1;
                    }
                    segs[n++] = (segment){a, b, labels[s]};
                }
                on = 0;
            }
        }
    }
    qsort(segs, n, sizeof(segment), order);
    *out = segs;
    *count = n;
    return 0;
}
void diar_result_free(diar_result *r) {
    free(r->segments);
    free(r->exclusive);
    free(r->chunk_counts);
    memset(r, 0, sizeof(*r));
}
int diar_run(const char *directory, const float *audio, size_t samples, int segment_only,
             diar_result *out, diar_cancel cancel, void *context) {
    memset(out, 0, sizeof(*out));
    int result = -1;
    float *masks = NULL, *emb = NULL, *train = NULL, *cent = NULL, *sums = NULL, *votes = NULL;
    float *window = NULL;
    uint32_t *normal = NULL, *exclusive = NULL;
    diar_checkpoint segmentation = {0}, embedding = {0};
    diar_plda *plda = NULL;
    char path[4096];
    if (!audio || !samples || samples > 16000U * 120U || !directory || (cancel && cancel(context)))
        goto done;
    if (snprintf(path, sizeof(path), "%s/segmentation-pytorch_model.bin", directory) >=
            (int)sizeof(path) ||
        diar_checkpoint_open(&segmentation, path))
        goto done;
    size_t chunks = samples < DIAR_SAMPLES ? 1
                                           : (samples - DIAR_SAMPLES) / 16000 + 1 +
                                                 ((samples - DIAR_SAMPLES) % 16000 != 0);
    masks = calloc(chunks * DIAR_FRAMES * 3, sizeof(float));
    emb = malloc(chunks * 3 * 256 * sizeof(float));
    train = malloc(chunks * 3 * 256 * sizeof(float));
    cent = malloc(chunks * 3 * 256 * sizeof(float));
    window = calloc(DIAR_SAMPLES, sizeof(float));
    if (!masks || !emb || !train || !cent || !window)
        goto done;
    for (size_t i = 0; i < chunks * 3 * 256; ++i)
        emb[i] = NAN;
    size_t train_n = 0;
    int any = 0;
    for (size_t c = 0; c < chunks; ++c) {
        float p[DIAR_FRAMES * 7];
        memset(window, 0, DIAR_SAMPLES * sizeof(float));
        size_t off = c * 16000, n = samples - off;
        if (n > DIAR_SAMPLES)
            n = DIAR_SAMPLES;
        memcpy(window, audio + off, n * sizeof(float));
        if (cancel && cancel(context))
            goto done;
        if (diar_segment(&segmentation, window, p)) {
            fprintf(stderr, "segmentation failed\n");
            goto done;
        }
        float *mask = masks + c * DIAR_FRAMES * 3;
        diar_powerset(p, mask, DIAR_FRAMES);
        /* Zero padded tail frames cannot describe speech outside the input. */
        int active = 0;
        for (int t = 0; t < DIAR_FRAMES; ++t)
            for (int s = 0; s < 3; ++s) {
                if (t * 270 + 495.5 >= n)
                    mask[t * 3 + s] = 0;
                active += mask[t * 3 + s] > 0;
            }
        any |= active > 0;
        if (segment_only || !active)
            continue;
        if (!embedding.mapping) {
            if (snprintf(path, sizeof(path), "%s/embedding-pytorch_model.bin", directory) >=
                    (int)sizeof(path) ||
                diar_checkpoint_open(&embedding, path))
                goto done;
        }
        float clean[DIAR_FRAMES * 3];
        int counts[3] = {0};
        for (int t = 0; t < DIAR_FRAMES; ++t) {
            int count = (int)(mask[t * 3] + mask[t * 3 + 1] + mask[t * 3 + 2]);
            for (int s = 0; s < 3; ++s) {
                clean[t * 3 + s] = count < 2 ? mask[t * 3 + s] : 0;
                counts[s] += (int)clean[t * 3 + s];
            }
        }
        for (int s = 0; s < 3; ++s)
            if (counts[s] <= 2)
                for (int t = 0; t < DIAR_FRAMES; ++t)
                    clean[t * 3 + s] = mask[t * 3 + s];
        if (cancel && cancel(context))
            goto done;
        if (diar_embed(&embedding, window, clean, emb + c * 3 * 256)) {
            fprintf(stderr, "embedding failed\n");
            goto done;
        }
        for (int s = 0; s < 3; ++s)
            if (counts[s] >= 0.2 * DIAR_FRAMES && isfinite(emb[(c * 3 + s) * 256])) {
                memcpy(train + train_n * 256, emb + (c * 3 + s) * 256, 256 * sizeof(float));
                ++train_n;
            }
    }
    if (segment_only) {
        out->chunks = chunks;
        out->chunk_counts = calloc(chunks * 3, sizeof(int));
        if (!out->chunk_counts)
            goto done;
        for (size_t c = 0; c < chunks; ++c)
            for (int t = 0; t < DIAR_FRAMES; ++t)
                for (int j = 0; j < 3; ++j)
                    out->chunk_counts[c * 3 + j] += (int)masks[(c * DIAR_FRAMES + t) * 3 + j];
        result = 0;
        goto done;
    }
    if (cancel && cancel(context))
        goto done;
    int k = 0;
    if (any) {
        if (!train_n) {
            fprintf(stderr, "insufficient clean speech to estimate speakers; no "
                            "diarization emitted\n");
            goto done;
        }
        plda = malloc(sizeof(*plda));
        char other[4096];
        if (!plda ||
            snprintf(path, sizeof(path), "%s/plda-xvec_transform.npz", directory) >=
                (int)sizeof(path) ||
            snprintf(other, sizeof(other), "%s/plda-plda.npz", directory) >= (int)sizeof(other) ||
            diar_plda_open(plda, path, other))
            goto done;
        k = diar_cluster(plda, train, train_n, cent);
        if (k < 1 || k > 32) {
            fprintf(stderr, "clustering failed or exceeded 32-speaker limit\n");
            goto done;
        }
    }
    size_t frames = (size_t)ceil(samples / 270.0) + DIAR_FRAMES;
    sums = calloc(frames * (size_t)(k ? k : 1), sizeof(float));
    votes = calloc(frames * 2, sizeof(float));
    normal = calloc(frames, sizeof(uint32_t));
    exclusive = calloc(frames, sizeof(uint32_t));
    if (!sums || !votes || !normal || !exclusive)
        goto done;
    for (size_t c = 0; c < chunks; ++c) {
        int labels[3] = {-1, -1, -1};
        float *mask = masks + c * DIAR_FRAMES * 3;
        int active = 0;
        for (int t = 0; t < DIAR_FRAMES * 3; ++t)
            active += mask[t] > 0;
        if (active && assign(emb + c * 3 * 256, mask, cent, k, labels))
            goto done;
        size_t start = (size_t)nearbyint(c * 16000.0 / 270);
        for (int t = 0; t < DIAR_FRAMES; ++t) {
            size_t at = start + t;
            votes[at * 2] += mask[t * 3] + mask[t * 3 + 1] + mask[t * 3 + 2];
            votes[at * 2 + 1] += 1;
            for (int s = 0; s < 3; ++s)
                if (labels[s] >= 0)
                    sums[at * k + labels[s]] += mask[t * 3 + s];
        }
    }
    int labels[32], next = 0;
    for (int i = 0; i < 32; ++i)
        labels[i] = -1;
    for (size_t t = 0; t < frames; ++t) {
        if ((t * 270 + 495.5) / 16000.0 >= samples / 16000.0)
            continue;
        int count = votes[t * 2 + 1] > 0 ? (int)nearbyint(votes[t * 2] / votes[t * 2 + 1]) : 0;
        if (count > k)
            count = k;
        uint32_t used = 0;
        for (int a = 0; a < count; ++a) {
            int best = -1;
            for (int s = 0; s < k; ++s)
                if (!((used >> s) & 1) && (best < 0 || sums[t * k + s] > sums[t * k + best]))
                    best = s;
            if (best < 0)
                break;
            used |= UINT32_C(1) << best;
            if (!a)
                exclusive[t] = UINT32_C(1) << best;
            if (labels[best] < 0)
                labels[best] = next++;
        }
        normal[t] = used;
    }
    out->speakers = next;
    out->training_embeddings = train_n;
    for (int i = 0; i < k; ++i)
        if (labels[i] >= 0)
            memcpy(out->centroids + labels[i] * 256, cent + i * 256, 256 * sizeof(float));
    if (intervals(normal, frames, k, labels, samples / 16000.0, &out->segments, &out->count) ||
        intervals(exclusive, frames, k, labels, samples / 16000.0, &out->exclusive,
                  &out->exclusive_count))
        goto done;
    result = 0;
done:
    free(window);
    diar_checkpoint_close(&segmentation);
    diar_checkpoint_close(&embedding);
    free(plda);
    free(masks);
    free(emb);
    free(train);
    free(cent);
    free(sums);
    free(votes);
    free(normal);
    free(exclusive);
    if (result)
        diar_result_free(out);
    return result;
}
