#define _POSIX_C_SOURCE 200809L
#include "../src/x86/whisper_turbo_w8a8.h"
#include "../src/x86/whisper_turbo_q8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
void wt_bench_encoder(int,const unsigned char *,const float *,size_t,size_t,size_t,float *);
static uint32_t rng=1701;
static uint32_t random_u32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv) {
    int threads=argc>1?atoi(argv[1]):8,which=argc>2?atoi(argv[2]):0;
    if(threads<1||threads>8||which<0||which>5)return 2;
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(threads);
#else
    threads=1;
#endif
    signed char a[128],b[128];
    for(int trial=0;trial<10000;trial++) {
        for(size_t i=0;i<128;i++) {
            a[i]=trial==0?-128:trial==1?127:(signed char)random_u32();
            b[i]=trial<2?a[i]:(signed char)random_u32();
        }
        int32_t ref=wt_i8_dot_scalar(a,b);
        if((wt_q8_has_avx2()&&wt_i8_dot_avx2(a,b)!=ref)||
           (wt_has_vnni()&&wt_i8_dot_vnni(a,b)!=ref)) {
            fprintf(stderr,"integer dot parity failed, trial %d\n",trial);return 1;
        }
    }
    const size_t cases[6][3]={{1,1280,1280},{1,1280,5120},{1,5120,1280},
        {1,1280,51866},{1500,1280,1280},{1500,1280,5120}};
    size_t m=cases[which][0],k=cases[which][1],n=cases[which][2],wb=n*(k/128)*130;
    unsigned char *w=malloc(wb);
    float *x=malloc(m*k*4),*ref=malloc(m*n*4),*out=malloc(m*n*4),*exact=malloc(m*n*4);
    if(!w||!x||!ref||!out||!exact)return 1;
    for(size_t g=0;g<wb/130;g++) {
        uint16_t s=(uint16_t)(0x3980+random_u32()%512);memcpy(w+130*g,&s,2);
        for(size_t j=0;j<128;j++)w[130*g+2+j]=(unsigned char)random_u32();
    }
    for(size_t i=0;i<m*k;i++)x[i]=((int)(random_u32()%20001)-10000)/10000.0f;
    wt_bench_encoder(wt_q8_has_avx512()?2:wt_q8_has_avx2()?1:0,w,x,m,k,n,ref);
    if(wt_w8a8_gemm(0,w,x,m,k,n,NULL,exact))return 1;
    const char *names[]={"scalar_int8","avx2_int8","avx512_vnni"};
    for(int mode=0;mode<3;mode++) {
        if((mode==1&&!wt_q8_has_avx2())||(mode==2&&!wt_has_vnni()))continue;
        for(int trial=0;trial<3;trial++) {
            double start=now();
            if(wt_w8a8_gemm(mode,w,x,m,k,n,NULL,out))return 1;
            double seconds=now()-start,err=0,norm=0,max=0;
            if(memcmp(out,exact,m*n*4)){fprintf(stderr,"GEMM integer parity failed\n");return 1;}
            for(size_t i=0;i<m*n;i++) {
                double e=(double)out[i]-ref[i];err+=e*e;norm+=(double)ref[i]*ref[i];
                if(fabs(e)>max)max=fabs(e);
            }
            double nrmse=sqrt(err/(norm+1e-30));
            if(!isfinite(nrmse)||nrmse>0.02)return 1;
            printf("{\"case\":%d,\"mode\":\"%s\",\"threads\":%d,\"trial\":%d,\"seconds\":%.9f,\"nrmse\":%.9g,\"max_abs_error\":%.9g,\"scratch_bytes\":%zu,\"integer_parity\":true}\n",
                which,names[mode],threads,trial,seconds,nrmse,max,m*k+m*k/128*4);
            fflush(stdout);
        }
    }
    free(w);free(x);free(ref);free(out);free(exact);return 0;
}
