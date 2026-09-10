#ifndef WT_INFERENCE_H
#define WT_INFERENCE_H
#include "api.h"
#include "whisper_turbo_image.h"
typedef struct {
    cllm_whisper_turbo_model model;
    unsigned threads;
} wt_engine;
int wt_transcribe(void *, const wt_request *, wt_result *, wt_error *, wt_cancel, void *);
#endif
