#include "api.h"
#include <stdlib.h>
#include <string.h>
static int digit(unsigned char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    return c == '+' ? 62 : c == '/' ? 63 : -1;
}
int wt_reference_wav(const unsigned char *url, size_t n, unsigned char **out, size_t *samples,
                     wt_error *e) {
    const char prefix[] = "data:audio/wav;base64,";
    *out = NULL;
    if (n <= sizeof(prefix) - 1 || n > 430000 || memcmp(url, prefix, sizeof(prefix) - 1))
        goto invalid;
    url += sizeof(prefix) - 1;
    n -= sizeof(prefix) - 1;
    if (n % 4)
        goto invalid;
    unsigned char *data = malloc(n / 4 * 3);
    if (!data)
        return wt_fail(e, 503, "Reference allocation failed.", NULL, "resource_exhausted");
    size_t bytes = 0;
    for (size_t i = 0; i < n; i += 4) {
        int a = digit(url[i]), b = digit(url[i + 1]);
        int c = url[i + 2] == '=' ? 0 : digit(url[i + 2]);
        int d = url[i + 3] == '=' ? 0 : digit(url[i + 3]);
        int padding = (url[i + 2] == '=') + (url[i + 3] == '=');
        if (a < 0 || b < 0 || c < 0 || d < 0 || (padding && i + 4 != n) ||
            (url[i + 2] == '=' && url[i + 3] != '=') || (padding == 2 && (b & 15)) ||
            (padding == 1 && (c & 3))) {
            free(data);
            goto invalid;
        }
        unsigned v = (unsigned)a << 18 | (unsigned)b << 12 | (unsigned)c << 6 | (unsigned)d;
        data[bytes++] = v >> 16;
        if (padding < 2)
            data[bytes++] = v >> 8;
        if (!padding)
            data[bytes++] = v;
    }
    const unsigned char *pcm;
    if (wt_wav(data, bytes, &pcm, samples, e) || *samples < 32000 || *samples > 160000) {
        free(data);
        goto invalid;
    }
    memmove(data, pcm, *samples * 2);
    *out = data;
    return 0;
invalid:
    return wt_fail(e, 400,
                   "References must be canonical base64 data:audio/wav URLs "
                   "containing 2–10 seconds of mono PCM16/16kHz audio.",
                   "known_speaker_references", "invalid_audio");
}
