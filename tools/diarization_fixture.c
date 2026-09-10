#define _POSIX_C_SOURCE 200809L
#include "audio.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int put16(FILE *f, unsigned v) {
    return fputc(v & 255, f) != EOF && fputc((v >> 8) & 255, f) != EOF ? 0 : -1;
}
static int put32(FILE *f, unsigned v) {
    return put16(f, v) || put16(f, v >> 16) ? -1 : 0;
}
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: diarization-fixture SPEAKER_A.wav SPEAKER_B.wav "
                        "NEW_OUTPUT.wav\nCreates 8s A / 1s silence / 8s B / 1s silence / 8s A.\n");
        return 2;
    }
    size_t na = 0, nb = 0;
    float *a = diar_wav_read(argv[1], &na), *b = diar_wav_read(argv[2], &nb);
    if (!a || !b || na < 128000 || nb < 128000) {
        free(a);
        free(b);
        return 1;
    }
    FILE *f = fopen(argv[3], "wbx");
    if (!f) {
        free(a);
        free(b);
        return 1;
    }
    unsigned samples = 26 * 16000;
    int bad = fwrite("RIFF", 1, 4, f) != 4 || put32(f, 36 + samples * 2) ||
              fwrite("WAVEfmt ", 1, 8, f) != 8 || put32(f, 16) || put16(f, 1) || put16(f, 1) ||
              put32(f, 16000) || put32(f, 32000) || put16(f, 2) || put16(f, 16) ||
              fwrite("data", 1, 4, f) != 4 || put32(f, samples * 2);
    for (unsigned i = 0; i < samples && !bad; ++i) {
        float v = i < 128000                  ? a[i]
                  : i >= 144000 && i < 272000 ? b[i - 144000]
                  : i >= 288000               ? a[i - 288000]
                                              : 0;
        int pcm = (int)lrintf(fmaxf(-1, fminf(32767.f / 32768, v)) * 32768);
        bad = put16(f, (unsigned)(uint16_t)(int16_t)pcm);
    }
    if (fclose(f))
        bad = 1;
    free(a);
    free(b);
    return bad ? 1 : 0;
}
