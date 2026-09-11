#include "search.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Quality/fallback criteria follow whisper.cpp (MIT), whisper_sequence_score
   and whisper_full_with_state. Token entropy uses natural logarithms, not the
   byte-compression ratio used by OpenAI Whisper. These are heuristics, not an
   accuracy guarantee. Unlike permissive final fallbacks, failed candidates are
   never returned merely because the retry budget is exhausted. */
double wt_token_entropy(const uint32_t *tokens, size_t count, uint32_t eot) {
    if ((!tokens && count) || count > WT_SEARCH_MAX_TEXT + 1) return NAN;
    uint32_t tail[32];
    size_t n = count + 1 < 32 ? count + 1 : 32;
    for (size_t i = 0; i + 1 < n; ++i) tail[i] = tokens[count - (n - 1) + i];
    tail[n - 1] = eot;
    double entropy = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t j = 0, frequency = 0;
        for (; j < i; ++j) if (tail[j] == tail[i]) break;
        if (j < i) continue;
        for (j = i; j < n; ++j) frequency += tail[j] == tail[i];
        double p = (double)frequency / n;
        entropy -= p * log(p);
    }
    return entropy;
}
int wt_decode_quality(const uint32_t *tokens, size_t count, uint32_t eot,
                      double sum_logprob, double no_speech_probability) {
    if ((!tokens && count) || count > WT_SEARCH_MAX_TEXT || !isfinite(sum_logprob) ||
        sum_logprob > 0 || !isfinite(no_speech_probability) ||
        no_speech_probability < 0 || no_speech_probability > 1) return WT_DECODE_RETRY;
    double average = sum_logprob / (count + 1);
    if (average < -1.0)
        return no_speech_probability > 0.6 ? WT_DECODE_SILENCE : WT_DECODE_RETRY;
    if (!count || (count + 1 > 32 && wt_token_entropy(tokens, count, eot) < 2.4))
        return WT_DECODE_RETRY;
    return WT_DECODE_ACCEPT;
}
static int distribution(const float *logits, size_t vocabulary, double scale,
                         double *maximum, double *sum) {
    if (!logits || !vocabulary || vocabulary > 51866 || !isfinite(scale) || scale <= 0) return -1;
    *maximum = -INFINITY; *sum = 0;
    for (size_t i = 0; i < vocabulary; ++i) {
        if (isnan(logits[i]) || logits[i] == INFINITY) return -1;
        if (logits[i] > *maximum) *maximum = logits[i];
    }
    if (!isfinite(*maximum)) return -1;
    for (size_t i = 0; i < vocabulary; ++i)
        *sum += exp(((double)logits[i] - *maximum) / scale);
    return !isfinite(*sum) || *sum <= 0 ? -1 : 0;
}
int wt_logits_logprob(const float *logits, size_t vocabulary, uint32_t token, double *logprob) {
    double maximum, sum;
    if (!logprob || token >= vocabulary || distribution(logits, vocabulary, 1, &maximum, &sum)) return -1;
    *logprob = ((double)logits[token] - maximum) - log(sum);
    return 0;
}
int wt_sample_token(const float *logits, size_t vocabulary, double temperature,
                    uint64_t *rng, uint32_t *token, double *logprob) {
    double maximum, sum, scale = temperature > 0 ? temperature : 1;
    if (!token || !logprob || !rng || !isfinite(temperature) || temperature < 0 || temperature > 1 ||
        distribution(logits, vocabulary, scale, &maximum, &sum)) return -1;
    uint32_t selected = 0;
    if (!temperature) {
        for (size_t i = 0; i < vocabulary; ++i)
            if (logits[i] == maximum) { selected = (uint32_t)i; break; }
    } else {
        /* SplitMix64: fixed request-local seeds, no process-global RNG state. */
        uint64_t z = (*rng += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        z ^= z >> 31;
        double remaining = (double)(z >> 11) * 0x1.0p-53 * sum;
        for (size_t i = 0; i < vocabulary; ++i) {
            if (!isfinite(logits[i])) continue;
            selected = (uint32_t)i;
            remaining -= exp(((double)logits[i] - maximum) / scale);
            if (remaining < 0) break;
        }
    }
    *token = selected;
    *logprob = ((double)logits[selected] - maximum) / scale - log(sum);
    return isfinite(*logprob) ? 0 : -1;
}

typedef struct {
    uint32_t tokens[WT_SEARCH_MAX_TEXT];
    size_t count;
    double score;
} hypothesis;
typedef struct { size_t parent; uint32_t token; double score; } candidate;

static int better(candidate a, candidate b) {
    return a.score > b.score || (a.score == b.score &&
        (a.parent < b.parent || (a.parent == b.parent && a.token < b.token)));
}

int wt_beam_search(size_t vocabulary, uint32_t eot, uint32_t start,
                   size_t maximum_text, wt_search_step step, wt_search_select select,
                   void *context, uint32_t *output, size_t *count) {
    if (!vocabulary || vocabulary > 51866 || eot >= vocabulary || start >= vocabulary ||
        !maximum_text || maximum_text > WT_SEARCH_MAX_TEXT || !step || !select || !output || !count)
        return -1;
    float *logits = malloc(vocabulary * sizeof(float));
    if (!logits) return -1;
    hypothesis live[WT_SEARCH_BEAMS] = {0}, next[WT_SEARCH_BEAMS], finished[WT_SEARCH_BEAMS];
    size_t active = 1, done = 0;
    int result = -1;
    for (size_t n = 0; n <= maximum_text && active; ++n) {
        candidate all[WT_SEARCH_BEAMS * (WT_SEARCH_BEAMS + 1)];
        size_t total = 0;
        for (size_t b = 0; b < active; ++b) {
            if (step(context, b, n ? live[b].tokens[n - 1] : start, n == 0, logits)) goto end;
            double maximum = -INFINITY, sum = 0;
            for (size_t t = 0; t < vocabulary; ++t) {
                if (isnan(logits[t]) || logits[t] == INFINITY) goto end;
                if (logits[t] > maximum) maximum = logits[t];
            }
            if (!isfinite(maximum)) goto end;
            candidate top[WT_SEARCH_BEAMS + 1];
            size_t ntop = 0;
            for (size_t t = 0; t < vocabulary; ++t) {
                if (!isfinite(logits[t])) continue;
                sum += exp(logits[t] - maximum);
                candidate c = {b, (uint32_t)t, logits[t]};
                size_t at = ntop;
                while (at && better(c, top[at - 1])) --at;
                if (at <= WT_SEARCH_BEAMS) {
                    size_t last = ntop < WT_SEARCH_BEAMS + 1 ? ntop++ : WT_SEARCH_BEAMS;
                    for (size_t j = last; j > at; --j) top[j] = top[j - 1];
                    top[at] = c;
                }
            }
            double normalizer = maximum + log(sum);
            for (size_t t = 0; t < ntop; ++t) {
                candidate c = top[t];
                c.score += live[b].score - normalizer;
                size_t at = total++;
                while (at && better(c, all[at - 1])) { all[at] = all[at - 1]; --at; }
                all[at] = c;
            }
        }
        size_t parents[WT_SEARCH_BEAMS], keep = 0;
        for (size_t i = 0; i < total; ++i) {
            candidate c = all[i];
            if (c.token == eot) {
                /* Blank first outputs are filtered by the caller, but never
                   accept one even if a callback supplies an unfiltered EOT. */
                if (!n) continue;
                hypothesis h = live[c.parent]; h.score = c.score;
                size_t at = done;
                while (at && h.score > finished[at - 1].score) --at;
                if (at < WT_SEARCH_BEAMS) {
                    size_t last = done < WT_SEARCH_BEAMS ? done++ : WT_SEARCH_BEAMS - 1;
                    for (size_t j = last; j > at; --j) finished[j] = finished[j - 1];
                    finished[at] = h;
                }
            } else if (n < maximum_text) {
                next[keep] = live[c.parent];
                next[keep].tokens[n] = c.token;
                next[keep].count = n + 1;
                next[keep].score = c.score;
                parents[keep++] = c.parent;
                if (keep == WT_SEARCH_BEAMS) break;
            }
        }
        if (done == WT_SEARCH_BEAMS || !keep) break;
        if (select(context, parents, keep)) goto end;
        memcpy(live, next, keep * sizeof(*live)); active = keep;
    }
    if (done) {
        size_t best = done;
        for (size_t i = 0; i < done; ++i) {
            if (wt_decode_quality(finished[i].tokens, finished[i].count, eot,
                                  finished[i].score, 0) != WT_DECODE_ACCEPT) continue;
            if (best == done || finished[i].score / (finished[i].count + 1) >
                finished[best].score / (finished[best].count + 1)) best = i;
        }
        if (best == done) goto end;
        memcpy(output, finished[best].tokens, finished[best].count * sizeof(*output));
        *count = finished[best].count;
        result = 0;
    }
end:
    free(logits);
    return result;
}
