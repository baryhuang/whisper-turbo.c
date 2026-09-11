#ifndef WT_W8A8_H
#define WT_W8A8_H
#include <stddef.h>
#include <stdint.h>
#define WT_W8A8_TILE_ROWS 16
#define WT_W8A8_MAX_K 5120
#define WT_W8A8_SCRATCH_BYTES (WT_W8A8_TILE_ROWS * WT_W8A8_MAX_K * sizeof(signed char) + \
                             WT_W8A8_TILE_ROWS * (WT_W8A8_MAX_K / 128) * sizeof(float))
int wt_has_vnni(void);
int32_t wt_i8_dot_scalar(const signed char *, const signed char *);
int32_t wt_i8_dot_avx2(const signed char *, const signed char *);
int32_t wt_i8_dot_vnni(const signed char *, const signed char *);
/* Experimental: quantizes activations per 128-element group using bounded
 * 16-row scratch, with k <= 5120. Returns -1 on invalid dimensions or nonfinite
 * input, with no output written. Reentrant; no retained heap allocation. */
int wt_w8a8_gemm(int mode, const unsigned char *, const float *, size_t, size_t,
                  size_t, const float *, float *);
#endif
