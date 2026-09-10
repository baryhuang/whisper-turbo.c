#ifndef DIAR_AUDIO_H
#define DIAR_AUDIO_H
#include <stddef.h>
/* Strict mono 16 kHz PCM16 reader. Caller frees result. Limit: 120 seconds. */
float *diar_wav_read(const char *path, size_t *samples);
#endif
