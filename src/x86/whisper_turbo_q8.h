#ifndef WHISPER_TURBO_X86_Q8_H
#define WHISPER_TURBO_X86_Q8_H
#include <stddef.h>
typedef float (*wt_group_dot_fn)(const unsigned char *, const float *);
int wt_q8_has_avx2(void);
int wt_q8_has_avx512(void);
void wt_q8_init(void);
float wt_q8_group_scalar(const unsigned char *, const float *);
float wt_q8_group_avx2(const unsigned char *, const float *);
float wt_q8_group_avx512(const unsigned char *, const float *);
float wt_q8_group_auto(const unsigned char *, const float *);
float wt_q8_row_auto(const unsigned char *, const float *, size_t);
void wt_q8_gemm_auto(const unsigned char *,const float *,size_t,size_t,size_t,const float *,float *);
#endif
