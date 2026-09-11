#include "alignment.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* C implementation of the cross-attention normalization / median-7 / DTW
   alignment method from OpenAI Whisper (MIT), whisper/timing.py. Capture uses
   the existing autoregressive pass, not a second teacher-forced inference. */
int wt_align(float *q, size_t positions, size_t tokens, size_t frames,
             size_t sf, size_t st, size_t *bounds) {
    if (!q || !bounds || !tokens || tokens + 5 != positions || positions > st ||
        st > 448 || !frames || frames > sf || sf > 1500)
        return -1;
    size_t rows = tokens + 1;
    float *matrix = calloc(rows * frames, sizeof(float));
    double *cost = malloc(2 * (frames + 1) * sizeof(double));
    unsigned char *trace = malloc(rows * frames);
    if (!matrix || !cost || !trace) {
        free(matrix); free(cost); free(trace); return -1;
    }
    for (size_t h = 0; h < 6; ++h) {
        float *head = q + h * st * sf;
        for (size_t t = 0; t < positions; ++t) {
            float *p = head + t * sf, maximum = -INFINITY;
            for (size_t f = 0; f < frames; ++f) {
                if (!isfinite(p[f])) goto fail;
                if (p[f] > maximum) maximum = p[f];
            }
            double sum = 0;
            for (size_t f = 0; f < frames; ++f) sum += p[f] = expf(p[f] - maximum);
            for (size_t f = 0; f < frames; ++f) p[f] /= (float)sum;
        }
        for (size_t f = 0; f < frames; ++f) {
            double sum = 0, square = 0;
            for (size_t t = 0; t < positions; ++t) {
                double v = head[t * sf + f]; sum += v; square += v * v;
            }
            double mean = sum / positions;
            double dev = sqrt(fmax(0, square / positions - mean * mean));
            for (size_t t = 0; t < positions; ++t)
                head[t * sf + f] = dev > 1e-12 ? (float)((head[t * sf + f] - mean) / dev) : 0;
        }
        for (size_t t = 0; t < rows; ++t)
            for (size_t f = 0; f < frames; ++f) {
                float sorted[7];
                size_t width = frames > 3 ? 7 : 1;
                for (size_t k = 0; k < width; ++k) {
                    long at = (long)f + (long)k - (long)(width / 2);
                    if (at < 0) at = -at;
                    if ((size_t)at >= frames) at = (long)(2 * frames - 2) - at;
                    sorted[k] = head[(t + 3) * sf + (size_t)at];
                    for (size_t j = k; j && sorted[j] < sorted[j - 1]; --j) {
                        float v = sorted[j]; sorted[j] = sorted[j - 1]; sorted[j - 1] = v;
                    }
                }
                matrix[t * frames + f] += sorted[width / 2] / 6;
            }
    }
    double *prev = cost, *cur = cost + frames + 1;
    prev[0] = 0;
    for (size_t j = 1; j <= frames; ++j) prev[j] = INFINITY;
    for (size_t i = 1; i <= rows; ++i) {
        cur[0] = INFINITY;
        for (size_t j = 1; j <= frames; ++j) {
            double a = prev[j - 1], b = prev[j], c = cur[j - 1];
            unsigned char dir = a < b && a < c ? 0 : b < a && b < c ? 1 : 2;
            cur[j] = (dir == 0 ? a : dir == 1 ? b : c) - matrix[(i - 1) * frames + j - 1];
            trace[(i - 1) * frames + j - 1] = dir;
        }
        double *swap = prev; prev = cur; cur = swap;
    }
    size_t i = rows, j = frames;
    while (i && j) {
        bounds[i - 1] = j - 1;
        unsigned char dir = trace[(i - 1) * frames + j - 1];
        if (dir != 2) --i;
        if (dir != 1) --j;
    }
    if (i || j) goto fail;
    free(matrix); free(cost); free(trace); return 0;
fail:
    free(matrix); free(cost); free(trace); return -1;
}

int wt_alignment_words(const unsigned char *text, size_t text_length,
                       const size_t *offsets, const size_t *bounds, size_t tokens,
                       size_t sample_offset, size_t samples, wt_word *words,
                       size_t capacity, size_t *count) {
    if (!text || !offsets || !bounds || !tokens || tokens > 448 || !words || !count ||
        *count > capacity || sample_offset >= samples || samples > WT_AUDIO_LIMIT)
        return -1;
    size_t used = samples - sample_offset;
    if (used > 480000) used = 480000;
    size_t frames = used / 320;
    if (!frames) frames = 1;
    for (size_t t = 0; t <= tokens; ++t) {
        if (offsets[t] > text_length || bounds[t] >= frames ||
            (t && (offsets[t] < offsets[t - 1] || bounds[t] < bounds[t - 1])))
            return -1;
    }
    const size_t first_word = *count;
    for (size_t first = 0; first < tokens;) {
        size_t last = first + 1;
        while (last < tokens) {
            unsigned char c = offsets[last] < text_length ? text[offsets[last]] : 0;
            if ((c == ' ' || c == '\n' || c == '\t') && bounds[last] > bounds[first]) break;
            ++last;
        }
        if (bounds[last] == bounds[first]) {
            /* DTW permits vertical steps: final tokens (often punctuation) can
               share the terminal boundary. Keep them in the preceding positive
               span of this window; do not fabricate a positive word duration. */
            if (*count == first_word) return -1;
            wt_word *previous = words + *count - 1;
            previous->length = offsets[last] - previous->offset;
            if (getenv("WHISPER_DIAGNOSTICS"))
                fprintf(stderr, "alignment: merged %zu collapsed trailing tokens at sample offset %zu\n",
                        last - first, sample_offset);
        } else {
            if (*count == capacity) return -1;
            words[(*count)++] = (wt_word){offsets[first], offsets[last] - offsets[first],
                sample_offset / 16000.0 + bounds[first] * 0.02,
                fmin(samples / 16000.0, sample_offset / 16000.0 + bounds[last] * 0.02)};
        }
        first = last;
    }
    return 0;
}

int wt_assign_speakers(wt_result *r, const diar_result *d, const char names[32][64],
                       const wt_request *req, wt_error *e) {
    if (!r->word_count) return 0;
    if (!d->exclusive_count)
        return wt_fail(e, 422, "Speech text has no diarization evidence.", "file", "alignment_failed");
    r->segments = calloc(WT_SEGMENT_LIMIT, sizeof(wt_segment));
    if (!r->segments) return wt_fail(e, 503, "Segment allocation failed.", NULL, "resource_exhausted");
    int labels[32];
    for (int k = 0; k < 32; ++k) labels[k] = -1;
    int next = 0, previous = -1;
    size_t segment_offset = 0;
    for (size_t i = 0; i < r->word_count; ++i) {
        const wt_word *w = r->words + i;
        double best = -1, distance = INFINITY;
        int speaker = -1;
        for (size_t k = 0; k < d->exclusive_count; ++k) {
            const diar_interval *s = d->exclusive + k;
            double overlap = fmax(0, fmin(w->end, s->end) - fmax(w->start, s->start));
            double mid = (w->start + w->end) / 2;
            double gap = fmax(0, fmax(s->start - mid, mid - s->end));
            if (overlap > best || (overlap == best && gap < distance)) {
                best = overlap; distance = gap; speaker = s->speaker;
            }
        }
        if (speaker < 0 || speaker >= 32 || w->offset > r->length || w->length > r->length - w->offset)
            return wt_fail(e, 422, "Invalid token alignment.", "file", "alignment_failed");
        if (speaker != previous) {
            if (r->segment_count == WT_SEGMENT_LIMIT)
                return wt_fail(e, 422, "Too many speaker turns.", "file", "output_limit_exceeded");
            wt_segment *s = r->segments + r->segment_count++;
            s->start = w->start;
            segment_offset = w->offset;
            if (names[speaker][0]) memcpy(s->speaker, names[speaker], sizeof(s->speaker));
            else {
                if (labels[speaker] < 0) {
                    for (;;) {
                        char label[64];
                        if (next < 26) snprintf(label, sizeof(label), "%c", 'A' + next);
                        else snprintf(label, sizeof(label), "A%c", 'A' + next - 26);
                        int collision = 0;
                        for (unsigned k = 0; k < req->name_count; ++k) collision |= !strcmp(label, req->names[k]);
                        labels[speaker] = next++;
                        if (!collision) break;
                    }
                }
                int label = labels[speaker];
                if (label < 26) snprintf(s->speaker, sizeof(s->speaker), "%c", 'A' + label);
                else snprintf(s->speaker, sizeof(s->speaker), "A%c", 'A' + label - 26);
            }
            previous = speaker;
        }
        wt_segment *s = r->segments + r->segment_count - 1;
        s->end = w->end;
        size_t n = w->offset + w->length - segment_offset;
        unsigned char *text = realloc(s->text, n + 1);
        if (!text) return wt_fail(e, 503, "Segment allocation failed.", NULL, "resource_exhausted");
        s->text = text;
        memcpy(text, r->text + segment_offset, n); text[n] = 0; s->length = n;
    }
    return 0;
}
