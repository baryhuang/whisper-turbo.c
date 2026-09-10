#include "whisper_turbo_attention.h"
#include "whisper_turbo_q8.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WT_X86 1
#else
#define WT_X86 0
#endif
static double dot_scalar(const float *q,const float *k) {
    double s=0;for(size_t c=0;c<64;c++)s+=(double)q[c]*k[c];return s;
}
static void context_scalar(const float *p,const float *v,size_t frames,size_t stride,float *out) {
    for(size_t c=0;c<64;c++) {
        double s=0;for(size_t f=0;f<frames;f++)s+=(double)p[f]*v[f*stride+c];
        out[c]=(float)s;
    }
}
#if WT_X86
__attribute__((target("avx2,fma")))
static double dot_avx2(const float *q,const float *k) {
    __m256d a=_mm256_setzero_pd(),b=a;
    for(size_t c=0;c<64;c+=8) {
        a=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(q+c)),_mm256_cvtps_pd(_mm_loadu_ps(k+c)),a);
        b=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(q+c+4)),_mm256_cvtps_pd(_mm_loadu_ps(k+c+4)),b);
    }
    a=_mm256_add_pd(a,b);__m128d s=_mm_add_pd(_mm256_castpd256_pd128(a),_mm256_extractf128_pd(a,1));
    return _mm_cvtsd_f64(_mm_hadd_pd(s,s));
}
__attribute__((target("avx512f,avx2,fma")))
static double dot_avx512(const float *q,const float *k) {
    __m512d a=_mm512_setzero_pd(),b=a;
    for(size_t c=0;c<64;c+=16) {
        a=_mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(q+c)),_mm512_cvtps_pd(_mm256_loadu_ps(k+c)),a);
        b=_mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(q+c+8)),_mm512_cvtps_pd(_mm256_loadu_ps(k+c+8)),b);
    }
    return _mm512_reduce_add_pd(_mm512_add_pd(a,b));
}
__attribute__((target("avx2,fma")))
static void context_avx2(const float *p,const float *v,size_t frames,size_t stride,float *out) {
    for(size_t c=0;c<64;c+=16) {
        __m256d a=_mm256_setzero_pd(),b=a,d=a,e=a;
        for(size_t f=0;f<frames;f++) {
            __m256d prob=_mm256_set1_pd(p[f]);const float *row=v+f*stride+c;
            a=_mm256_fmadd_pd(prob,_mm256_cvtps_pd(_mm_loadu_ps(row)),a);
            b=_mm256_fmadd_pd(prob,_mm256_cvtps_pd(_mm_loadu_ps(row+4)),b);
            d=_mm256_fmadd_pd(prob,_mm256_cvtps_pd(_mm_loadu_ps(row+8)),d);
            e=_mm256_fmadd_pd(prob,_mm256_cvtps_pd(_mm_loadu_ps(row+12)),e);
        }
        _mm_storeu_ps(out+c,_mm256_cvtpd_ps(a));_mm_storeu_ps(out+c+4,_mm256_cvtpd_ps(b));
        _mm_storeu_ps(out+c+8,_mm256_cvtpd_ps(d));_mm_storeu_ps(out+c+12,_mm256_cvtpd_ps(e));
    }
}
__attribute__((target("avx512f,avx2,fma")))
static void context_avx512(const float *p,const float *v,size_t frames,size_t stride,float *out) {
    for(size_t c=0;c<64;c+=32) {
        __m512d a=_mm512_setzero_pd(),b=a,d=a,e=a;
        for(size_t f=0;f<frames;f++) {
            __m512d prob=_mm512_set1_pd(p[f]);const float *row=v+f*stride+c;
            a=_mm512_fmadd_pd(prob,_mm512_cvtps_pd(_mm256_loadu_ps(row)),a);
            b=_mm512_fmadd_pd(prob,_mm512_cvtps_pd(_mm256_loadu_ps(row+8)),b);
            d=_mm512_fmadd_pd(prob,_mm512_cvtps_pd(_mm256_loadu_ps(row+16)),d);
            e=_mm512_fmadd_pd(prob,_mm512_cvtps_pd(_mm256_loadu_ps(row+24)),e);
        }
        _mm256_storeu_ps(out+c,_mm512_cvtpd_ps(a));_mm256_storeu_ps(out+c+8,_mm512_cvtpd_ps(b));
        _mm256_storeu_ps(out+c+16,_mm512_cvtpd_ps(d));_mm256_storeu_ps(out+c+24,_mm512_cvtpd_ps(e));
    }
}
#endif
void wt_attention(size_t frames,size_t state,size_t heads,float *q,const float *k,
                    const float *v,float *scores,float *out) {
    /* Whisper Turbo has 20 heads of width 64. The hook is target-specific. */
    if(!heads||state/heads!=64||state%heads)abort();
    double (*dot)(const float *,const float *)=dot_scalar;
    void (*context)(const float *,const float *,size_t,size_t,float *)=context_scalar;
#if WT_X86
    const char *mode=getenv("WHISPER_SIMD");
    if(!mode||strcmp(mode,"scalar")) {
        if(wt_q8_has_avx2()){dot=dot_avx2;context=context_avx2;}
        if(wt_q8_has_avx512()&&(!mode||strcmp(mode,"avx2"))){dot=dot_avx512;context=context_avx512;}
    }
#endif
    const float scale=1.0f/sqrtf(64.0f);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for(size_t h=0;h<heads;h++)for(size_t row=0;row<frames;row++) {
        float *p=scores;
#ifdef _OPENMP
        p+=(size_t)omp_get_thread_num()*frames;
#endif
        float max=-INFINITY;double denominator=0;
        for(size_t f=0;f<frames;f++) {
            p[f]=(float)dot(q+row*state+h*64,k+f*state+h*64)*scale;
            if(p[f]>max)max=p[f];
        }
        for(size_t f=0;f<frames;f++){p[f]=expf(p[f]-max);denominator+=p[f];}
        for(size_t f=0;f<frames;f++)p[f]=(float)((double)p[f]/denominator);
        context(p,v+h*64,frames,state,out+row*state+h*64);
    }
}
