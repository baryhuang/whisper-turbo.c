#include "audio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned le16(const unsigned char *p) {
    return p[0] | p[1] << 8;
}
static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
float *diar_wav_read(const char *path, size_t *samples) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    unsigned char h[16];
    float *audio = NULL;
    long size;
    if (fseek(f, 0, SEEK_END) || (size = ftell(f)) < 12 ||
        (unsigned long)size > WT_MAX_AUDIO_SAMPLES * 2U + 1024U * 1024U ||
        fseek(f, 0, SEEK_SET) || fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) ||
        memcmp(h + 8, "WAVE", 4) || (uint64_t)le32(h + 4) + 8 != (uint64_t)size)
        goto done;
    int fmt = 0, seen = 0;
    long data = 0;
    size_t n = 0;
    while (ftell(f) < size) {
        long pos = ftell(f);
        if (size - pos < 8 || fread(h, 1, 8, f) != 8)
            goto done;
        uint32_t bytes = le32(h + 4);
        if ((uint64_t)pos + 8 + bytes + (bytes & 1) > (uint64_t)size)
            goto done;
        if (!memcmp(h, "fmt ", 4)) {
            if (fmt || bytes < 16 || fread(h, 1, 16, f) != 16 || le16(h) != 1 || le16(h + 2) != 1 ||
                le32(h + 4) != 16000 || le32(h + 8) != 32000 || le16(h + 12) != 2 ||
                le16(h + 14) != 16)
                goto done;
            fmt = 1;
        } else if (!memcmp(h, "data", 4)) {
            if (seen || !bytes || (bytes & 1) || bytes > WT_MAX_AUDIO_SAMPLES * 2U)
                goto done;
            seen = 1;
            data = pos + 8;
            n = bytes / 2;
        }
        if (fseek(f, pos + 8 + bytes + (bytes & 1), SEEK_SET))
            goto done;
    }
    if (!fmt || !seen || fseek(f, data, SEEK_SET))
        goto done;
    audio = malloc(n * sizeof(float));
    if (!audio)
        goto done;
    for (size_t i = 0; i < n; ++i) {
        unsigned char b[2];
        if (fread(b, 1, 2, f) != 2) {
            free(audio);
            audio = NULL;
            goto done;
        }
        audio[i] = (int16_t)le16(b) / 32768.f;
    }
    *samples = n;
done:
    fclose(f);
    return audio;
}
