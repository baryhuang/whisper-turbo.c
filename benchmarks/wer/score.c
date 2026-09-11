#define _POSIX_C_SOURCE 200809L
#include "score.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unicase.h>
#include <unictype.h>
#include <uninorm.h>
#include <unistr.h>

char *wer_normalize(const char *text) {
    if (!text || strlen(text) > 262144) return NULL;
    size_t n = strlen(text), length = 0;
    if (u8_check((const uint8_t *)text, n)) return NULL;
    /* libunistring is implemented in C. NULL language requests Unicode's
       locale-independent, whole-string lowercasing, followed by NFKD. */
    uint8_t *norm = u8_tolower((const uint8_t *)text, n, NULL, UNINORM_NFKD, NULL, &length);
    if (!norm) return NULL;
    char *output = malloc(length + 1);
    if (!output) { free(norm); return NULL; }
    size_t used = 0; int space = 0;
    for (size_t at = 0; at < length;) {
        ucs4_t cp; int bytes = u8_mbtouc(&cp, norm + at, length - at);
        if (bytes <= 0) { free(norm); free(output); return NULL; }
        size_t begin = at; at += (size_t)bytes;
        if (uc_is_property_white_space(cp) || (cp >= 0x1c && cp <= 0x1f)) { space = used > 0; continue; }
        if (!(uc_is_general_category(cp, UC_LETTER) || uc_is_general_category(cp, UC_NUMBER) || cp == '_')) continue;
        if (space) output[used++] = ' ';
        space = 0; memcpy(output + used, norm + begin, (size_t)bytes); used += (size_t)bytes;
    }
    output[used] = 0; free(norm); return output;
}
static size_t words(char *text, char **out) {
    size_t n = 0; char *save = NULL;
    for (char *p = strtok_r(text, " ", &save); p; p = strtok_r(NULL, " ", &save)) {
        if (n == 4096) return SIZE_MAX;
        out[n++] = p;
    }
    return n;
}
typedef struct { size_t cost, s, d, i; } cell;
int wer_score(const char *reference, const char *hypothesis, wer_counts *out) {
    if (!out) return -1;
    char *r = wer_normalize(reference), *h = wer_normalize(hypothesis);
    char *rw[4096], *hw[4096]; cell *a = NULL, *b = NULL; int rc = -1;
    if (!r || !h) goto done;
    size_t nr = words(r, rw), nh = words(h, hw);
    if (nr == SIZE_MAX || nh == SIZE_MAX) goto done;
    a = calloc(nh + 1, sizeof(*a)); b = calloc(nh + 1, sizeof(*b));
    if (!a || !b) goto done;
    for (size_t j = 0; j <= nh; ++j) a[j] = (cell){j, 0, 0, j};
    for (size_t i = 1; i <= nr; ++i) {
        b[0] = (cell){i, 0, i, 0};
        for (size_t j = 1; j <= nh; ++j) {
            cell c = a[j - 1];
            if (strcmp(rw[i - 1], hw[j - 1])) { ++c.cost; ++c.s; }
            if (a[j].cost + 1 < c.cost) { c = a[j]; ++c.cost; ++c.d; }
            if (b[j - 1].cost + 1 < c.cost) { c = b[j - 1]; ++c.cost; ++c.i; }
            b[j] = c;
        }
        cell *swap = a; a = b; b = swap;
    }
    *out = (wer_counts){nr, nh, a[nh].s, a[nh].d, a[nh].i}; rc = 0;
done:
    free(r); free(h); free(a); free(b); return rc;
}
