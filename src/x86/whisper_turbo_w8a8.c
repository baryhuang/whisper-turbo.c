#include "whisper_turbo_w8a8.h"
#include "whisper_turbo_q8.h"
#include "../generic/whisper_turbo_quant.h"
#include <math.h>
#include <stdlib.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WT_X86 1
#else
#define WT_X86 0
#endif
int wt_has_vnni(void) {
#if WT_X86
    return wt_q8_has_avx512() && __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}
int32_t wt_i8_dot_scalar(const signed char *w,const signed char *x) {
    int32_t sum=0;
    for(size_t i=0;i<128;i++)sum+=(int32_t)w[i]*x[i];
    return sum;
}
#if WT_X86
__attribute__((target("avx2")))
int32_t wt_i8_dot_avx2(const signed char *w,const signed char *x) {
    __m256i sum=_mm256_setzero_si256();
    for(size_t i=0;i<128;i+=16) {
        __m256i a=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(w+i)));
        __m256i b=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(x+i)));
        /* Widen before multiplying: no saturating 16-bit byte-dot sums. */
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(a,b));
    }
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(sum),_mm256_extracti128_si256(sum,1));
    s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}
__attribute__((target("avx512f,avx512bw,avx512vnni")))
int32_t wt_i8_dot_vnni(const signed char *w,const signed char *x) {
    __m512i sum=_mm512_setzero_si512(),correction=sum;
    const __m512i shift=_mm512_set1_epi8((char)128);
    for(size_t i=0;i<128;i+=64) {
        __m512i a=_mm512_loadu_si512((const void *)(w+i));
        __m512i b=_mm512_xor_si512(_mm512_loadu_si512((const void *)(x+i)),shift);
        sum=_mm512_dpbusd_epi32(sum,b,a);
        correction=_mm512_dpbusd_epi32(correction,shift,a);
    }
    return _mm512_reduce_add_epi32(_mm512_sub_epi32(sum,correction));
}
#else
int32_t wt_i8_dot_avx2(const signed char *w,const signed char *x){return wt_i8_dot_scalar(w,x);}
int32_t wt_i8_dot_vnni(const signed char *w,const signed char *x){return wt_i8_dot_scalar(w,x);}
#endif
int wt_w8a8_gemm(int mode,const unsigned char *w,const float *x,size_t rows,
                  size_t k,size_t n,const float *bias,float *y) {
    /* Cache-sized row batches reuse the same bounded workspace. No per-layer
       malloc/free and no second resident model or unbounded thread-local cache.
       5120 is the largest input width of the supported Turbo model. */
    enum { TILE = WT_W8A8_TILE_ROWS, MAX_K = WT_W8A8_MAX_K };
    if(!w||!x||!y||!rows||!k||k%128||k>MAX_K||!n||
       rows>SIZE_MAX/k||rows>SIZE_MAX/n||n>SIZE_MAX/(k/128)/130)return -1;
    size_t groups=k/128;
    signed char q[TILE*MAX_K];
    float scale[TILE*MAX_K/128];
    /* Validate before writing any output, including errors in later tiles. */
    int bad=0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:bad) schedule(static) if(rows*k >= 32768)
#endif
    for(size_t i=0;i<rows*k;i++)bad |= !isfinite(x[i]);
    if(bad)return -1;
    int32_t (*dot)(const signed char *,const signed char *)=wt_i8_dot_scalar;
    if(mode>=1&&wt_q8_has_avx2())dot=wt_i8_dot_avx2;
    if(mode>=2&&wt_has_vnni())dot=wt_i8_dot_vnni;
    for(size_t first=0;first<rows;first+=TILE) {
        size_t count=rows-first<TILE?rows-first:TILE;
        for(size_t g=0;g<count*groups;g++) {
            float max=0;
            for(size_t i=0;i<128;i++)max=fmaxf(max,fabsf(x[first*k+g*128+i]));
            float s=max/127.0f;scale[g]=s;
            for(size_t i=0;i<128;i++) {
                long z=s>0?lrintf(x[first*k+g*128+i]/s):0;
                if(z>127)z=127;
                if(z< -127)z=-127;
                q[g*128+i]=(signed char)z;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(size_t col=0;col<n;col++) {
            const unsigned char *wr=w+col*groups*130;
            for(size_t row=0;row<count;row++) {
                float sum=bias?bias[col]:0;
                for(size_t g=0;g<groups;g++) {
                    const signed char *ww=(const signed char *)(const void *)(wr+g*130+2);
                    sum+=cllm_whisper_turbo_bf16(wr+g*130)*scale[row*groups+g]*
                         (float)dot(ww,q+row*k+g*128);
                }
                y[(first+row)*n+col]=sum;
            }
        }
    }
    return 0;
}
