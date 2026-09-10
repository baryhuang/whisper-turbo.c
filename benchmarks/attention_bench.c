#define _POSIX_C_SOURCE 200809L
#include "../src/x86/whisper_turbo_attention.h"
#include "../src/x86/whisper_turbo_q8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
void wt_reference_attention(size_t,size_t,size_t,float *,const float *,const float *,float *,float *);
static uint32_t rng=1701;
static float random_float(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return ((int)(rng%20001)-10000)/5000.0f;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv) {
    size_t frames=argc>1?(size_t)atoi(argv[1]):64;
    int threads=argc>2?atoi(argv[2]):8;
    if(!frames||frames>1500||threads<1||threads>8)return 2;
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(threads);
#endif
    size_t count=frames*1280;
    float *q=malloc(count*4),*k=malloc(count*4),*v=malloc(count*4),*s=malloc(frames*8*4);
    float *ref=malloc(count*4),*out=malloc(count*4);
    if(!q||!k||!v||!s||!ref||!out)return 1;
    for(size_t i=0;i<count;i++){q[i]=random_float();k[i]=random_float();v[i]=random_float();}
    double start=now();wt_reference_attention(frames,1280,20,q,k,v,s,ref);
    printf("{\"mode\":\"generic\",\"frames\":%zu,\"threads\":%d,\"seconds\":%.9f}\n",frames,threads,now()-start);
    const char *names[]={"scalar","avx2","avx512"};
    for(int mode=0;mode<3;mode++) {
        if((mode==1&&!wt_q8_has_avx2())||(mode==2&&!wt_q8_has_avx512()))continue;
        setenv("WHISPER_SIMD",names[mode],1);
        start=now();wt_attention(frames,1280,20,q,k,v,s,out);double sec=now()-start,max=0;
        for(size_t i=0;i<count;i++) {
            double e=fabs((double)out[i]-ref[i]);if(e>max)max=e;
            if(!isfinite(out[i])||e>2e-6+2e-5*fabs(ref[i]))return 1;
        }
        printf("{\"mode\":\"%s\",\"frames\":%zu,\"threads\":%d,\"seconds\":%.9f,\"max_abs_error\":%.9g}\n",names[mode],frames,threads,sec,max);
    }
    free(q);free(k);free(v);free(s);free(ref);free(out);return 0;
}
