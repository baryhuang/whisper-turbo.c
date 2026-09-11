#ifndef DIAR_AUDIO_H
#define DIAR_AUDIO_H
#include <stddef.h>
#include "../audio_limits.h"
/* Strict mono 16 kHz PCM16 reader. Caller frees result. Limit: 300 seconds. */
float *diar_wav_read(const char *path, size_t *samples);
#endif
