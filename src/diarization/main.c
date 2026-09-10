#define _POSIX_C_SOURCE 200809L
#include "audio.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void emit(const diar_interval *s, size_t n) {
    putchar('[');
    for (size_t i = 0; i < n; ++i)
        printf("%s{\"start\":%.6f,\"end\":%.6f,\"speaker\":\"SPEAKER_%02d\"}", i ? "," : "",
               s[i].start, s[i].end, s[i].speaker);
    putchar(']');
}
int main(int argc, char **argv) {
    int only = argc == 4 && !strcmp(argv[3], "--segmentation-only");
    if (argc != 3 && !only) {
        fprintf(stderr, "usage: diarize-community MODEL_DIRECTORY AUDIO.wav "
                        "[--segmentation-only]\n");
        return 2;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    size_t samples = 0;
    float *audio = diar_wav_read(argv[2], &samples);
    diar_result r;
    if (diar_run(argv[1], audio, samples, only, &r, NULL, NULL)) {
        free(audio);
        fprintf(stderr, "Diarization failed; check model files and mono "
                        "PCM16/16kHz WAV (up to 120 seconds).\n");
        return 1;
    }
    free(audio);
    if (only) {
        printf("{\"local_segmentation_only\":true,\"chunks\":[");
        for (size_t c = 0; c < r.chunks; ++c)
            printf("%s{\"start\":%zu,\"speaker_frames\":[%d,%d,%d]}", c ? "," : "", c,
                   r.chunk_counts[c * 3], r.chunk_counts[c * 3 + 1], r.chunk_counts[c * 3 + 2]);
        printf("]}\n");
    } else {
        printf("{\"model\":\"pyannote/"
               "speaker-diarization-community-1\",\"implementation\":\"experimental-"
               "c\",\"audio_seconds\":%.6f,\"num_speakers\":%d,\"diarization\":",
               samples / 16000.0, r.speakers);
        emit(r.segments, r.count);
        printf(",\"exclusive_diarization\":");
        emit(r.exclusive, r.exclusive_count);
        printf("}\n");
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    fprintf(stderr, "result speakers=%d training_embeddings=%zu elapsed_seconds=%.3f\n", r.speakers,
            r.training_embeddings,
            end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9);
    diar_result_free(&r);
    return ferror(stdout) ? 1 : 0;
}
