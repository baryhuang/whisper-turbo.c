#include "inference.h"
#include "pipeline.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned asr_calls, diar_calls;
int wt_transcribe_speech_pcm(wt_engine *engine, const unsigned char *pcm, size_t samples,
                              const char *language, wt_result *r, wt_error *e, wt_cancel cancel, void *context,
                              const diar_result *speech) {
    (void)engine; (void)pcm; (void)language; (void)e; (void)cancel; (void)context;
    assert(samples == 26 * 16000 && diar_calls == 1 && speech->exclusive_count == 3); ++asr_calls;
    r->text = malloc(15); assert(r->text); memcpy(r->text,"Hello Hi Again",15);
    r->length=14; r->duration=26; r->word_count=3; r->asr_windows=1;
    r->words=calloc(3,sizeof(wt_word)); assert(r->words);
    r->words[0]=(wt_word){0,5,1,2}; r->words[1]=(wt_word){5,3,11,12}; r->words[2]=(wt_word){8,6,21,22};
    return 0;
}
int wt_transcribe_pcm(wt_engine *a,const unsigned char *b,size_t c,const char *d,wt_result *e,
                      wt_error *f,wt_cancel g,void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    assert(!"Diarized pipeline must not retranscribe speaker crops"); return -1;
}
int diar_run(const char *directory,const float *audio,size_t samples,int only,diar_result *d,
              diar_cancel cancel,void *context) {
    (void)directory;(void)audio;(void)cancel;(void)context;
    assert(asr_calls==0 && samples==26*16000 && !only); ++diar_calls;
    d->exclusive=calloc(3,sizeof(diar_interval)); assert(d->exclusive);
    d->exclusive[0]=(diar_interval){0,8,0}; d->exclusive[1]=(diar_interval){9,17,1}; d->exclusive[2]=(diar_interval){18,26,0};
    d->exclusive_count=3;d->speakers=2;return 0;
}
void diar_result_free(diar_result *d) {free(d->exclusive);memset(d,0,sizeof(*d));}
static void le(unsigned char *p,unsigned n){for(int i=0;i<4;++i)p[i]=(unsigned char)(n>>(i*8));}
int main(void) {
    size_t bytes=44+26*32000;
    unsigned char *wav=calloc(bytes,1);assert(wav);
    memcpy(wav,"RIFF",4);le(wav+4,(unsigned)bytes-8);memcpy(wav+8,"WAVEfmt ",8);
    le(wav+16,16);wav[20]=1;wav[22]=1;le(wav+24,16000);le(wav+28,32000);wav[32]=2;wav[34]=16;
    memcpy(wav+36,"data",4);le(wav+40,26*32000);wav[44]=1;
    wt_request req={.file=wav,.file_size=bytes,.diarize=1,.diarized_json=1};
    wt_engine engine={.threads=1,.diarization_directory="test-only"};wt_result result={0};wt_error error={0};
    assert(!wt_transcribe(&engine,&req,&result,&error,NULL,NULL));
    assert(asr_calls==1 && diar_calls==1 && result.asr_windows==1 && result.diarization_passes==1);
    assert(result.segment_count==3 && !strcmp((char *)result.text,"Hello Hi Again"));
    assert(!strcmp(result.segments[0].speaker,"A") && !strcmp(result.segments[1].speaker,"B") && !strcmp(result.segments[2].speaker,"A"));
    wt_result_free(&result);free(wav);
    puts("C pipeline test passed: one full-recording ASR call, one diarizer call, no crop retranscription.");
    return 0;
}
