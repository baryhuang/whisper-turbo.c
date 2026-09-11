#ifndef WT_SEARCH_H
#define WT_SEARCH_H
#include <stddef.h>
#include <stdint.h>

enum { WT_SEARCH_BEAMS = 3, WT_SEARCH_MAX_TEXT = 443 };
enum { WT_DECODE_RETRY = 0, WT_DECODE_ACCEPT = 1, WT_DECODE_SILENCE = 2 };
double wt_token_entropy(const uint32_t *tokens, size_t count, uint32_t eot);
int wt_decode_quality(const uint32_t *tokens, size_t count, uint32_t eot,
                      double sum_logprob, double no_speech_probability);
int wt_logits_logprob(const float *logits, size_t vocabulary, uint32_t token, double *logprob);
int wt_sample_token(const float *logits, size_t vocabulary, double temperature,
                    uint64_t *rng, uint32_t *token, double *logprob);
/* step consumes token in a beam's cache and fills already-filtered logits.
   select reorders those updated caches for the next generation. */
typedef int (*wt_search_step)(void *, size_t beam, uint32_t token, int first, float *logits);
typedef int (*wt_search_select)(void *, const size_t *parents, size_t count);
/* Only naturally EOT-terminated hypotheses are eligible. Returns -1 on failure;
   output/count are untouched unless a complete hypothesis is selected. */
int wt_beam_search(size_t vocabulary, uint32_t eot, uint32_t start,
                   size_t maximum_text, wt_search_step step, wt_search_select select,
                   void *context, uint32_t *output, size_t *count);
#endif
