#ifndef WT_ALIGNMENT_H
#define WT_ALIGNMENT_H
#include "api.h"
#include "pipeline.h"
/* In-place normalization of captured QK scores, followed by median filtering
   and monotonic DTW. boundaries has text_tokens+1 entries (20ms frame indices). */
int wt_align(float *scores, size_t positions, size_t text_tokens, size_t frames,
             size_t stride_frames, size_t stride_tokens, size_t *boundaries);
/* Append whole-word groups from monotonic token boundaries. Tokens collapsed
   onto an existing group's end are kept with that group, without inventing time. */
int wt_alignment_words(const unsigned char *text, size_t text_length,
                       const size_t *offsets, const size_t *boundaries, size_t tokens,
                       size_t sample_offset, size_t samples, wt_word *words,
                       size_t capacity, size_t *count);
int wt_assign_speakers(wt_result *, const diar_result *, const char names[32][64],
                       const wt_request *, wt_error *);
int wt_speech_window_active(const diar_result *, size_t sample_offset, size_t used);
size_t wt_language_window_offset(const diar_result *, size_t samples);
int wt_filter_speech_words(wt_result *, const diar_result *, wt_error *);
#endif
