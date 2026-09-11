#include "whisper_turbo_q8.h"
#include "whisper_turbo_w8a8.h"
#include "../generic/whisper_turbo_quant.h"
#include <stdlib.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WT_X86 1
#else
#define WT_X86 0
#endif
int wt_q8_has_avx2(void) {
#if WT_X86
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
int wt_q8_has_avx512(void) {
#if WT_X86
    return wt_q8_has_avx2() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    return 0;
#endif
}
float wt_q8_group_scalar(const unsigned char *p, const float *x) {
    const signed char *w = (const signed char *)(const void *)p;
    float sum=0;
    for (size_t i=0;i<128;i++) sum+=x[i]*(float)w[i];
    return sum;
}
#if WT_X86
__attribute__((target("avx2,fma")))
float wt_q8_group_avx2(const unsigned char *p, const float *x) {
    __m256 a=_mm256_setzero_ps(), b=a;
    for (size_t i=0;i<128;i+=16) {
        __m128i q=_mm_loadu_si128((const __m128i *)(const void *)(p+i));
        __m256 lo=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
        __m256 hi=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q,8)));
        a=_mm256_fmadd_ps(lo,_mm256_loadu_ps(x+i),a);
        b=_mm256_fmadd_ps(hi,_mm256_loadu_ps(x+i+8),b);
    }
    a=_mm256_add_ps(a,b);
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}
__attribute__((target("avx512f,avx512bw,avx2,fma")))
float wt_q8_group_avx512(const unsigned char *p,const float *x) {
    __m512 a=_mm512_setzero_ps(), b=a;
    for(size_t i=0;i<128;i+=32) {
        __m512 lo=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(p+i))));
        __m512 hi=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(p+i+16))));
        a=_mm512_fmadd_ps(lo,_mm512_loadu_ps(x+i),a);
        b=_mm512_fmadd_ps(hi,_mm512_loadu_ps(x+i+16),b);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a,b));
}
#else
float wt_q8_group_avx2(const unsigned char *p,const float *x){return wt_q8_group_scalar(p,x);}
float wt_q8_group_avx512(const unsigned char *p,const float *x){return wt_q8_group_scalar(p,x);}
#endif
/* Initialize on the main thread before starting inference workers. */
static wt_group_dot_fn selected=wt_q8_group_scalar;
void wt_q8_init(void) {
    const char *mode=getenv("WHISPER_SIMD");
    selected=wt_q8_group_scalar;
    if(!mode || strcmp(mode,"scalar")) {
        if(wt_q8_has_avx2()) selected=wt_q8_group_avx2;
        if(wt_q8_has_avx512() && (!mode || strcmp(mode,"avx2"))) selected=wt_q8_group_avx512;
    }
}
float wt_q8_group_auto(const unsigned char *p,const float *x){return selected(p,x);}
float wt_q8_row_auto(const unsigned char *p,const float *x,size_t n) {
    float s=0;
    for(size_t g=0;g<n/128;g++)s+=cllm_whisper_turbo_bf16(p+130*g)*selected(p+130*g+2,x+128*g);
    return s;
}
int wt_decoder_q8_gemm(const unsigned char *w,const float *x,size_t rows,size_t k,
                       size_t n,const float *bias,float *y) {
    const char *activations=getenv("WHISPER_DECODER_ACTIVATIONS");
    if(!activations || strcmp(activations,"int8"))return -1;
    const char *simd=getenv("WHISPER_SIMD");
    int mode=simd&&!strcmp(simd,"scalar")?0:simd&&!strcmp(simd,"avx2")?1:2;
    return wt_w8a8_gemm(mode,w,x,rows,k,n,bias,y);
}
void wt_q8_gemm_auto(const unsigned char *w,const float *x,size_t rows,size_t k,
                       size_t n,const float *bias,float *y) {
    const char *activations=getenv("WHISPER_ACTIVATIONS");
    if(activations&&!strcmp(activations,"int8")) {
        const char *simd=getenv("WHISPER_SIMD");
        int mode=simd&&!strcmp(simd,"scalar")?0:simd&&!strcmp(simd,"avx2")?1:2;
        if(!wt_w8a8_gemm(mode,w,x,rows,k,n,bias,y))return;
        /* Memory pressure: bounded FP32-activation path requires no scratch. */
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t col=0;col<n;col++)for(size_t row=0;row<rows;row++) {
        float sum=bias?bias[col]:0;
        const unsigned char *wr=w+col*(k/128)*130;
        for(size_t g=0;g<k/128;g++)sum+=cllm_whisper_turbo_bf16(wr+g*130)*
            selected(wr+g*130+2,x+row*k+g*128);
        y[row*n+col]=sum;
    }
}
