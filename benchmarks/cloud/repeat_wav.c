/* C-only stress fixture: repeat a PCM16/16kHz/mono clip to exactly 30 seconds.
 * This is synthetic benchmark audio, not an independent accuracy corpus. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint32_t le32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void put32(unsigned char *p,uint32_t n){for(int i=0;i<4;i++)p[i]=(unsigned char)(n>>(i*8));}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    FILE *f=fopen(argv[1],"rb");unsigned char h[44],*pcm=NULL;size_t bytes=0;
    if(!f||fread(h,1,12,f)!=12||memcmp(h,"RIFF",4)||memcmp(h+8,"WAVE",4))return 2;
    int format=0;
    while(fread(h,1,8,f)==8) {
        uint32_t n=le32(h+4);if(n>16*1024*1024)return 2;
        if(!memcmp(h,"fmt ",4)) {
            if(n<16||n>40||fread(h,1,n,f)!=n)return 2;
            format=h[0]==1&&h[1]==0&&h[2]==1&&h[3]==0&&le32(h+4)==16000&&h[14]==16&&h[15]==0;
        }else if(!memcmp(h,"data",4)) {
            if(pcm||!n||(n&1))return 2;
            pcm=malloc(n);if(!pcm||fread(pcm,1,n,f)!=n)return 2;bytes=n;
        }else if(fseek(f,n,SEEK_CUR))return 2;
        if((n&1)&&fseek(f,1,SEEK_CUR))return 2;
    }
    fclose(f);if(!format||!pcm)return 2;
    unsigned char header[44]={'R','I','F','F',0,0,0,0,'W','A','V','E','f','m','t',' ',16,0,0,0,1,0,1,0};
    put32(header+4,960036);put32(header+24,16000);put32(header+28,32000);
    header[32]=2;header[34]=16;memcpy(header+36,"data",4);put32(header+40,960000);
    f=fopen(argv[2],"wbx");if(!f||fwrite(header,1,44,f)!=44)return 2;
    for(size_t pos=0;pos<960000;) {
        size_t n=960000-pos;if(n>bytes)n=bytes;
        if(fwrite(pcm,1,n,f)!=n)return 2;
        pos+=n;
    }
    free(pcm);return fclose(f)?2:0;
}
