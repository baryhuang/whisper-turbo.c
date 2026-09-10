#ifndef WT_INFERENCE_H
#define WT_INFERENCE_H
#include "api.h"
#include "whisper_turbo_image.h"
typedef struct {
    cllm_whisper_turbo_model model;
    unsigned threads;
    const char *diarization_directory;
} wt_engine;
int wt_transcribe(void *, const wt_request *, wt_result *, wt_error *, wt_cancel, void *);
int wt_transcribe_pcm(wt_engine *, const unsigned char *, size_t, const char *, wt_result *,
                      wt_error *, wt_cancel, void *);
int wt_transcribe_aligned_pcm(wt_engine *, const unsigned char *, size_t, const char *, wt_result *,
                              wt_error *, wt_cancel, void *);
#endif
