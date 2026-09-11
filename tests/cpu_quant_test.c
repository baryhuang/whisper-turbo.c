#define _POSIX_C_SOURCE 200809L
#include "../src/x86/whisper_turbo_q8.h"
#include "../src/x86/whisper_turbo_w8a8.h"
#include "../src/generic/whisper_turbo_quant.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(size_t rows, size_t k, size_t n) {
    size_t groups=k/128;
    unsigned char *w=malloc(n*groups*130);
    float *x=malloc(rows*k*sizeof(float)), *y=malloc(rows*n*sizeof(float));
    float *reference=calloc(rows*n,sizeof(float)), *bias=malloc(n*sizeof(float));
    assert(w && x && y && reference && bias);
    for(size_t col=0;col<n;col++) {
        bias[col]=(float)col/17;
        for(size_t g=0;g<groups;g++) {
            uint16_t bits=0x3c80;
            memcpy(w+(col*groups+g)*130,&bits,2);
            for(size_t i=0;i<128;i++)
                w[(col*groups+g)*130+2+i]=(unsigned char)(signed char)((int)((i+col+g)%255)-127);
        }
    }
    for(size_t i=0;i<rows*k;i++)x[i]=(float)((int)(i%29)-14)/7;
    /* Includes a zero activation group and negative extrema. */
    memset(x,0,128*sizeof(float));
    for(size_t row=0;row<rows;row++)for(size_t col=0;col<n;col++) {
        float sum=bias[col];
        for(size_t g=0;g<groups;g++) {
            float peak=0;
            for(size_t i=0;i<128;i++)peak=fmaxf(peak,fabsf(x[row*k+g*128+i]));
            float scale=peak/127;
            int32_t dot=0;
            for(size_t i=0;i<128;i++) {
                int q=scale>0?(int)lrintf(x[row*k+g*128+i]/scale):0;
                const signed char *weights=(const signed char *)(w+(col*groups+g)*130+2);
                dot+=q*weights[i];
            }
            sum+=cllm_whisper_turbo_bf16(w+(col*groups+g)*130)*scale*(float)dot;
        }
        reference[row*n+col]=sum;
    }
    for(int mode=0;mode<3;mode++) {
        assert(!wt_w8a8_gemm(mode,w,x,rows,k,n,bias,y));
        assert(!memcmp(y,reference,rows*n*sizeof(float)));
    }
    assert(!setenv("WHISPER_DECODER_ACTIVATIONS","int8",1));
    assert(!wt_decoder_q8_gemm(w,x,rows,k,n,bias,y));
    assert(!memcmp(y,reference,rows*n*sizeof(float)));
    assert(!unsetenv("WHISPER_DECODER_ACTIVATIONS"));
    y[0]=123;
    assert(wt_decoder_q8_gemm(w,x,rows,k,n,bias,y)==-1 && y[0]==123);
    for(size_t i=0;i<rows*n;i++)y[i]=123;
    x[rows*k-1]=NAN;
    assert(wt_w8a8_gemm(2,w,x,rows,k,n,bias,y)==-1);
    for(size_t i=0;i<rows*n;i++)assert(y[i]==123);
    assert(wt_w8a8_gemm(2,w,x,rows,129,n,bias,y)==-1);
    assert(wt_w8a8_gemm(2,w,x,rows,5248,n,bias,y)==-1);
    assert(wt_w8a8_gemm(2,w,x,SIZE_MAX,128,n,bias,y)==-1);
    free(w);free(x);free(y);free(reference);free(bias);
}
int main(void) {
    check(1,128,1);
    check(17,256,7);
    check(33,1280,17);
    check(1,5120,19);
    puts("CPU INT8 checks passed: independent reference, SIMD parity, tile tails, bias, zero groups, opt-in and atomic input rejection.");
    return 0;
}
