#include "audio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int put(FILE *f, unsigned n, int bytes) {
    for (int i=0;i<bytes;++i) if (fputc((n>>(8*i))&255,f)==EOF) return -1;
    return 0;
}
int main(int argc,char **argv) {
    if(argc!=5){fprintf(stderr,"usage: wav-slice INPUT.wav START_SECONDS LENGTH_SECONDS NEW_OUTPUT.wav\n");return 2;}
    char *end;double start=strtod(argv[2],&end);if(*end||!isfinite(start)||start<0||start>120)return 2;
    double length=strtod(argv[3],&end);if(*end||!isfinite(length)||length<=0||length>120)return 2;
    size_t count=0;float *audio=diar_wav_read(argv[1],&count);if(!audio)return 2;
    size_t first=(size_t)llround(start*16000),n=(size_t)llround(length*16000);
    if(!n||first>count||n>count-first){free(audio);return 2;}
    FILE *f=fopen(argv[4],"wbx");if(!f){free(audio);return 2;}
    int bad=fwrite("RIFF",1,4,f)!=4||put(f,36+(unsigned)n*2,4)||fwrite("WAVEfmt ",1,8,f)!=8||put(f,16,4)||put(f,1,2)||put(f,1,2)||put(f,16000,4)||put(f,32000,4)||put(f,2,2)||put(f,16,2)||fwrite("data",1,4,f)!=4||put(f,(unsigned)n*2,4);
    for(size_t i=0;i<n&&!bad;++i)bad=put(f,(unsigned)(int)lrintf(audio[first+i]*32768),2);
    free(audio);bad|=fclose(f)!=0;return bad?1:0;
}
