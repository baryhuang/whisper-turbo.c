#ifndef WT_W8A8_H
#define WT_W8A8_H
#include <stddef.h>
#include <stdint.h>
int wt_has_vnni(void);
int32_t wt_i8_dot_scalar(const signed char *, const signed char *);
int32_t wt_i8_dot_avx2(const signed char *, const signed char *);
int32_t wt_i8_dot_vnni(const signed char *, const signed char *);
/* Experimental: quantizes activations per 128-element group. Returns -1 on
 * allocation failure or nonfinite input, with no output written. */
int wt_w8a8_gemm(int mode, const unsigned char *, const float *, size_t, size_t,
                  size_t, const float *, float *);
#endif
