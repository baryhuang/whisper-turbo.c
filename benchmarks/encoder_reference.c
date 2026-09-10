/* Benchmark the actual inherited loop, substituting only its group dot. */
#include "../src/generic/whisper_turbo_encoder.c"
#include "../src/x86/whisper_turbo_q8.h"
void wt_bench_encoder(int mode, const unsigned char *w, const float *x,
                      size_t rows, size_t k, size_t n, float *y)
{
    wt_group_dot_fn dot = mode == 2 ? wt_q8_group_avx512 :
                          mode == 1 ? wt_q8_group_avx2 : whisper_turbo_q8_group_dot;
    linear_grouped(x,rows,k,w,CLLM_WHISPER_TURBO_Q8_RECORD,dot,NULL,n,y);
}
