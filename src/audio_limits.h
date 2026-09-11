#ifndef WT_AUDIO_LIMITS_H
#define WT_AUDIO_LIMITS_H
/* Shared HTTP, CLI, and diarization bounds; ASR workspace stays windowed. */
#define WT_MAX_AUDIO_SECONDS 300U
#define WT_MAX_AUDIO_SAMPLES (16000U * WT_MAX_AUDIO_SECONDS)
#endif
