#ifndef DIAR_NETWORK_H
#define DIAR_NETWORK_H
#include "checkpoint.h"
#define DIAR_SAMPLES 160000
#define DIAR_FRAMES 589
#define DIAR_EMBEDDING 256
float diar_dot(const float *a, const float *b, size_t n);
/* Input is exactly ten seconds, 16 kHz mono; output is frame-major
 * seven-way powerset probabilities. No speaker identity is global here. */
int diar_segment(const diar_checkpoint *model, const float *audio, float *probabilities);
/* Three segmentation masks, frame-major. Shared ResNet computation followed
 * by separate weighted pooling; output contains three 256-D embeddings.
 * An insufficient mask produces NaNs (never a made-up speaker vector). */
int diar_embed(const diar_checkpoint *model, const float *audio, const float *masks,
               float *embeddings);
int diar_fbank(const float *audio, size_t samples, float *features);
void diar_powerset(const float *probabilities, float *masks, size_t frames);
#endif
