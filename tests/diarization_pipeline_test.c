/* Synthetic model outputs only: no private audio or checkpoint dependency. */
#include "pipeline.h"
#include "network.h"
#include "cluster.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int scenario, chunk, current;
int diar_checkpoint_open(diar_checkpoint *c, const char *path) {
    (void)path; c->mapping = c; return 0;
}
void diar_checkpoint_close(diar_checkpoint *c) { c->mapping = NULL; }
int diar_segment(const diar_checkpoint *m, const float *a, float *p) {
    (void)m; (void)a; memset(p,0,DIAR_FRAMES*7*sizeof(float)); current=chunk++; return 0;
}
void diar_powerset(const float *p, float *mask, size_t frames) {
    (void)p; memset(mask,0,frames*3*sizeof(float));
    if (current < 2 && scenario != 2) {
        for(size_t t=0;t<frames;++t) mask[t*3+(scenario==1 && t>=frames/2)]=1;
    } else {
        /* A short positive mask missed by embedding temporal downsampling. */
        mask[10*3]=1;
    }
}
int diar_embed(const diar_checkpoint *m, const float *a, const float *mask, float *e) {
    (void)m; (void)a; (void)mask;
    for(int i=0;i<3*256;++i)e[i]=NAN;
    if(scenario==4 && current>=2) memset(e,0,3*256*sizeof(float));
    if(current<2 && scenario!=2) {
        for(int s=0;s<(scenario==1?2:1);++s) {
            memset(e+s*256,0,256*sizeof(float)); e[s*256+s]=1;
        }
    }
    return 0;
}
int diar_plda_open(diar_plda *p,const char *a,const char *b) {
    (void)p;(void)a;(void)b;return 0;
}
int diar_cluster(const diar_plda *p,const float *e,size_t n,float *cent) {
    (void)p;
    int k=scenario==1?2:1;
    assert(n==(size_t)k*2); memcpy(cent,e,(size_t)k*256*sizeof(float));
    if(scenario==3) cent[0]=NAN;
    return k;
}
int main(void) {
    float *audio=calloc(192000,sizeof(float)); assert(audio);
    for(scenario=0;scenario<5;++scenario) {
        chunk=0; diar_result r={0};
        int rc=diar_run("synthetic",audio,192000,0,&r,NULL,NULL);
        if(scenario==2 || scenario==3) assert(rc && !r.exclusive_count);
        else {
            assert(!rc && r.speakers==(scenario==1?2:1) && r.exclusive_count>0);
            for(size_t i=0;i<r.exclusive_count;++i) {
                assert(r.exclusive[i].start<r.exclusive[i].end);
                assert(r.exclusive[i].speaker>=0 && r.exclusive[i].speaker<r.speakers);
            }
        }
        diar_result_free(&r);
    }
    free(audio);
    puts("Diarization pipeline checks passed: missing local embeddings, one/two speakers, no global evidence.");
    return 0;
}
