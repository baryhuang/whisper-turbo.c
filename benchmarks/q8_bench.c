#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/x86/whisper_turbo_q8.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
void wt_bench_encoder(int,const unsigned char *,const float *,size_t,size_t,size_t,float *);
static uint32_t rng=1701;
static uint32_t random_u32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmp(const void *a,const void *b){double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);}
int main(int argc,char **argv) {
    int threads=argc>1?atoi(argv[1]):8,which=argc>2?atoi(argv[2]):0;
    if(threads<1||threads>8||which<0||which>5)return 2;
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(threads);
#else
    threads=1;
#endif
    const size_t cases[6][3]={{1,1280,1280},{1,1280,5120},{1,5120,1280},{1,1280,51866},{1500,1280,1280},{1500,1280,5120}};
    size_t m=cases[which][0],k=cases[which][1],n=cases[which][2],wb=n*(k/128)*130;
    unsigned char *w=malloc(wb);float *x=malloc(m*k*4),*ref=malloc(m*n*4),*out=malloc(m*n*4);
    if(!w||!x||!ref||!out)return 1;
    for(size_t g=0;g<wb/130;g++) {
        uint16_t scale=(uint16_t)(0x3980+(random_u32()%512));memcpy(w+130*g,&scale,2);
        for(size_t j=0;j<128;j++)w[130*g+2+j]=(unsigned char)random_u32();
    }
    for(size_t i=0;i<m*k;i++)x[i]=((int)(random_u32()%20001)-10000)/10000.0f;
    wt_bench_encoder(0,w,x,m,k,n,ref);
    int available[3]={1,wt_q8_has_avx2(),wt_q8_has_avx512()},passed=1;
    double errors[3]={0},times[3][5]={{0}};
    for(int mode=1;mode<3;mode++)if(available[mode]) {
        wt_bench_encoder(mode,w,x,m,k,n,out);
        for(size_t i=0;i<m*n;i++) {
            double e=fabs((double)out[i]-ref[i]);if(e>errors[mode])errors[mode]=e;
            if(!isfinite(out[i])||e>5e-4+5e-5*fabs(ref[i]))passed=0;
        }
    }
    if(!passed){fprintf(stderr,"numerical parity failed\n");return 1;}
    /* Rotate execution order to reduce warmup and shared-CPU scheduling bias. */
    for(int trial=0;trial<5;trial++)for(int slot=0;slot<3;slot++) {
        int mode=(trial+slot)%3;if(!available[mode])continue;
        double start=now();wt_bench_encoder(mode,w,x,m,k,n,out);times[mode][trial]=now()-start;
    }
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    unsigned long long rss=(unsigned long long)usage.ru_maxrss;
#ifndef __APPLE__
    rss*=1024;
#endif
    const char *names[]={"generic","avx2","avx512"};
    for(int mode=0;mode<3;mode++)if(available[mode]) {
        double sorted[5];memcpy(sorted,times[mode],sizeof(sorted));qsort(sorted,5,sizeof(double),cmp);
        printf("{\"case\":%d,\"rows\":%zu,\"input\":%zu,\"output\":%zu,\"precision\":\"int8_weights_fp32_activations\",\"mode\":\"%s\",\"threads\":%d,\"median_seconds\":%.9f,\"max_abs_error\":%.9g,\"peak_rss_bytes\":%llu,\"trials_seconds\":[%.9f,%.9f,%.9f,%.9f,%.9f]}\n",
               which,m,k,n,names[mode],threads,sorted[2],errors[mode],rss,times[mode][0],times[mode][1],times[mode][2],times[mode][3],times[mode][4]);
    }
    free(w);free(x);free(ref);free(out);return 0;
}
