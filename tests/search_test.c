#include "../src/server/search.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { VOCAB = 8, END = 7, START = 6 };
typedef struct {
    unsigned path[WT_SEARCH_BEAMS], depth[WT_SEARCH_BEAMS];
    size_t calls, selections, cancel_after;
    int mode, cancel;
} fixture;
static int step(void *opaque, size_t beam, uint32_t token, int first, float *logits) {
    fixture *f = opaque;
    if (f->cancel || (f->cancel_after && f->calls >= f->cancel_after)) return -1;
    ++f->calls;
    assert(beam < WT_SEARCH_BEAMS && (first ? token == START : token != START));
    if (first) { f->path[beam] = 0; f->depth[beam] = 0; }
    else { if (!f->path[beam]) f->path[beam] = token; ++f->depth[beam]; }
    for (int i = 0; i < VOCAB; ++i) logits[i] = -INFINITY;
    if (f->mode == 1) { logits[1] = 0; return 0; } /* No EOT anywhere. */
    if (f->mode == 2) return 0; /* All suppressed. */
    if (f->mode == 3) { logits[1] = NAN; return 0; }
    if (f->mode == 4) { logits[1] = INFINITY; return 0; }
    if (f->mode == 5) { logits[END] = 0; return 0; } /* Invalid blank. */
    if (f->mode == 6) {
        logits[f->depth[beam] < 40 ? 1 : END] = 0;
        return 0; /* EOT does not rescue a repetitive candidate. */
    }
    if (first) { logits[1] = 0.1f; logits[2] = 0; }
    else if (f->path[beam] == 1) logits[1] = 0; /* Greedy never stops. */
    else if (f->depth[beam] < 3) logits[3] = 0;
    else logits[END] = 0; /* Alternative naturally terminates. */
    return 0;
}
static int select_parents(void *opaque, const size_t *parents, size_t count) {
    fixture *f = opaque, saved = *f;
    ++f->selections;
    for (size_t i = 0; i < count; ++i) {
        assert(parents[i] < WT_SEARCH_BEAMS);
        f->path[i] = saved.path[parents[i]];
        f->depth[i] = saved.depth[parents[i]];
    }
    return 0;
}
int main(void) {
    uint32_t repeated[80], diverse[80];
    for (size_t i = 0; i < 80; ++i) { repeated[i] = (uint32_t)(i % 4); diverse[i] = (uint32_t)i; }
    assert(wt_token_entropy(repeated, 80, 100) < 2.4);
    assert(wt_token_entropy(diverse, 80, 100) > 3.4);
    assert(wt_decode_quality(repeated, 80, 100, -10, 0) == WT_DECODE_RETRY);
    assert(wt_decode_quality(diverse, 80, 100, -10, 0) == WT_DECODE_ACCEPT);
    assert(wt_decode_quality(diverse, 80, 100, -82, 0.6) == WT_DECODE_RETRY);
    assert(wt_decode_quality(diverse, 80, 100, -82, 0.61) == WT_DECODE_SILENCE);
    assert(wt_decode_quality(diverse, 80, 100, -81, 0.99) == WT_DECODE_ACCEPT);
    assert(wt_decode_quality(NULL, 1, 100, -1, 0) == WT_DECODE_RETRY);
    assert(wt_decode_quality(diverse, 80, 100, NAN, 0) == WT_DECODE_RETRY);
    assert(wt_decode_quality(diverse, 80, 100, -1, NAN) == WT_DECODE_RETRY);
    assert(wt_decode_quality(diverse, 80, 100, 1, 0) == WT_DECODE_RETRY);
    assert(wt_decode_quality(diverse, WT_SEARCH_MAX_TEXT + 1, 100, -1, 0) == WT_DECODE_RETRY);
    assert(wt_decode_quality(NULL, 0, 100, -2, 0.7) == WT_DECODE_SILENCE);
    assert(wt_decode_quality(NULL, 0, 100, -0.5, 0.7) == WT_DECODE_RETRY);
    float logits[] = {0, 0, -INFINITY, 0};
    uint32_t chosen; double logprob;
    uint64_t rng = 1;
    assert(!wt_sample_token(logits, 4, 0, &rng, &chosen, &logprob));
    assert(chosen == 0 && fabs(logprob + log(3)) < 1e-12 && rng == 1);
    assert(!wt_logits_logprob(logits, 4, 3, &logprob) && fabs(exp(logprob) - 1.0 / 3) < 1e-12);
    assert(!wt_logits_logprob(logits, 4, 2, &logprob) && logprob == -INFINITY);
    for (unsigned stage = 1; stage <= 5; ++stage) {
        uint64_t a = 42, b = 42;
        unsigned seen = 0;
        for (unsigned i = 0; i < 100; ++i) {
            uint32_t other; double other_logprob;
            assert(!wt_sample_token(logits, 4, stage / 5.0, &a, &chosen, &logprob));
            assert(!wt_sample_token(logits, 4, stage / 5.0, &b, &other, &other_logprob));
            assert(chosen == other && logprob == other_logprob && chosen != 2);
            seen |= 1U << chosen;
        }
        assert(seen == 11); /* Sampling explores all unmasked alternatives. */
    }
    float invalid[] = {-INFINITY, -INFINITY};
    assert(wt_sample_token(invalid, 2, 0.2, &rng, &chosen, &logprob));
    invalid[0] = NAN;
    assert(wt_sample_token(invalid, 2, 0.2, &rng, &chosen, &logprob));
    invalid[0] = INFINITY;
    assert(wt_sample_token(invalid, 2, 0.2, &rng, &chosen, &logprob));
    assert(wt_sample_token(logits, 4, NAN, &rng, &chosen, &logprob));
    assert(wt_sample_token(logits, 4, 1.1, &rng, &chosen, &logprob));
    assert(wt_sample_token(logits, 4, -0.1, &rng, &chosen, &logprob));
    float extremes[] = {-1e30f, 1e30f, -INFINITY};
    assert(!wt_sample_token(extremes, 3, 0.2, &rng, &chosen, &logprob));
    assert(chosen == 1 && logprob == 0);
    float shift_a[] = {2, 3, -INFINITY}, shift_b[] = {102, 103, -INFINITY};
    for (unsigned i = 0; i < 100; ++i) {
        uint64_t a = i, b = i; uint32_t other; double other_logprob;
        assert(!wt_sample_token(shift_a, 3, 0.4, &a, &chosen, &logprob));
        assert(!wt_sample_token(shift_b, 3, 0.4, &b, &other, &other_logprob));
        assert(chosen == other && logprob == other_logprob);
    }
    fixture f = {0};
    uint32_t output[WT_SEARCH_MAX_TEXT]; size_t count = 99;
    assert(!wt_beam_search(VOCAB, END, START, 7, step, select_parents, &f, output, &count));
    assert(count == 3 && output[0] == 2 && output[1] == 3 && output[2] == 3);
    uint32_t expected[3]; memcpy(expected, output, sizeof(expected));
    f = (fixture){0};
    assert(!wt_beam_search(VOCAB, END, START, 3, step, select_parents, &f, output, &count));
    assert(count == 3 && !memcmp(expected, output, sizeof(expected))); /* EOT at final allowed step. */
    f = (fixture){0}; count = 99; output[0] = 999;
    assert(wt_beam_search(VOCAB, END, START, 2, step, select_parents, &f, output, &count));
    assert(count == 99 && output[0] == 999); /* Never return unfinished beam. */
    for (int mode = 1; mode <= 5; ++mode) {
        f = (fixture){.mode=mode}; count = 99; output[0] = 999;
        assert(wt_beam_search(VOCAB, END, START, 7, step, select_parents, &f, output, &count));
        assert(count == 99 && output[0] == 999);
        assert(f.calls <= 8 * WT_SEARCH_BEAMS);
    }
    f = (fixture){.cancel=1};
    assert(wt_beam_search(VOCAB, END, START, 7, step, select_parents, &f, output, &count));
    assert(wt_beam_search(VOCAB, END, START, WT_SEARCH_MAX_TEXT + 1, step, select_parents, &f, output, &count));
    assert(wt_beam_search(VOCAB, END, START, 0, step, select_parents, &f, output, &count));
    f = (fixture){.mode=6}; count = 99; output[0] = 999;
    assert(wt_beam_search(VOCAB, END, START, 60, step, select_parents, &f, output, &count));
    assert(count == 99 && output[0] == 999);
    f = (fixture){.mode=1};
    assert(wt_beam_search(VOCAB, END, START, WT_SEARCH_MAX_TEXT, step, select_parents, &f, output, &count));
    assert(f.calls == WT_SEARCH_MAX_TEXT + 1);
    f = (fixture){.cancel_after=3}; count = 99; output[0] = 999;
    assert(wt_beam_search(VOCAB, END, START, 7, step, select_parents, &f, output, &count));
    assert(f.calls == 3 && count == 99 && output[0] == 999);
    puts("C search tests passed: quality gates, no-speech coupling, bounded beam search, natural EOT, stable sampling, cancellation and invalid logits.");
}
