#ifndef WT_ALIGNMENT_H
#define WT_ALIGNMENT_H
#include "api.h"
#include "pipeline.h"
/* In-place normalization of captured QK scores, followed by median filtering
   and monotonic DTW. boundaries has text_tokens+1 entries (20ms frame indices). */
int wt_align(float *scores, size_t positions, size_t text_tokens, size_t frames,
             size_t stride_frames, size_t stride_tokens, size_t *boundaries);
int wt_assign_speakers(wt_result *, const diar_result *, const char names[32][64],
                       const wt_request *, wt_error *);
#endif
