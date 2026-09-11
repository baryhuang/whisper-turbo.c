#include "api.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int wt_fail(wt_error *e, int status, const char *message, const char *param, const char *code) {
    *e = (wt_error){status, message, param, code};
    return -1;
}
/* Parse a MIME parameter list without interpreting filenames as paths. */
static int parameters(const char *s, const char *kind, char *name, size_t cap, int *filename) {
    size_t k = strlen(kind), used = 0;
    int found = 0, seen_filename = 0;
    if (strncasecmp(s, kind, k))
        return -1;
    s += k;
    while (*s == ' ' || *s == '\t')
        ++s;
    while (*s) {
        if (*s++ != ';')
            return -1;
        while (*s == ' ' || *s == '\t')
            ++s;
        const char *key = s;
        while (isalnum((unsigned char)*s) || *s == '-' || *s == '*' || *s == '_')
            ++s;
        size_t key_len = (size_t)(s - key);
        if (!key_len)
            return -1;
        while (*s == ' ' || *s == '\t')
            ++s;
        if (*s++ != '=')
            return -1;
        while (*s == ' ' || *s == '\t')
            ++s;
        int capture = (filename ? key_len == 4 && !strncasecmp(key, "name", 4)
                                : key_len == 8 && !strncasecmp(key, "boundary", 8));
        int file = filename && key_len == 8 && !strncasecmp(key, "filename", 8);
        if ((capture && found++) || (file && seen_filename++))
            return -1;
        used = 0;
        int quoted = *s == '"';
        if (quoted)
            ++s;
        while (*s && (quoted ? *s != '"' : *s != ';' && *s != ' ' && *s != '\t')) {
            unsigned char c = (unsigned char)*s++;
            if (quoted && c == '\\') {
                if (!*s)
                    return -1;
                c = (unsigned char)*s++;
            }
            if (c < 32 || c == 127)
                return -1;
            if (capture) {
                if (used + 1 >= cap)
                    return -1;
                name[used++] = (char)c;
            }
        }
        if (quoted && *s++ != '"')
            return -1;
        if (capture)
            name[used] = 0;
        while (*s == ' ' || *s == '\t')
            ++s;
    }
    if (filename)
        *filename = seen_filename;
    return found == 1 ? 0 : -1;
}
int wt_boundary(const char *type, char out[71]) {
    if (parameters(type, "multipart/form-data", out, 71, NULL) || !out[0])
        return -1;
    size_t n = strlen(out);
    if (out[n - 1] == ' ')
        return -1;
    for (size_t i = 0; i < n; ++i)
        if (!isalnum((unsigned char)out[i]) && !strchr("'()+_,-./:=? ", out[i]))
            return -1;
    return 0;
}
/* Linear-time binary search: adversarial repeated boundary prefixes stay
 * bounded. */
static const unsigned char *find(const unsigned char *s, size_t n, const char *key, size_t k) {
    size_t prefix[80] = {0}, matched = 0;
    if (!k || k > sizeof(prefix) / sizeof(*prefix))
        return NULL;
    for (size_t i = 1, j = 0; i < k; ++i) {
        while (j && key[i] != key[j])
            j = prefix[j - 1];
        if (key[i] == key[j])
            ++j;
        prefix[i] = j;
    }
    for (size_t i = 0; i < n; ++i) {
        while (matched && s[i] != (unsigned char)key[matched])
            matched = prefix[matched - 1];
        if (s[i] == (unsigned char)key[matched])
            ++matched;
        if (matched == k)
            return s + i + 1 - k;
    }
    return NULL;
}
int wt_multipart(const unsigned char *body, size_t n, const char *boundary, wt_request *r,
                 wt_error *e) {
    char marker[76];
    size_t b = strlen(boundary);
    if (!b || b > 70 || n > WT_UPLOAD_LIMIT)
        goto malformed;
    snprintf(marker, sizeof(marker), "\r\n--%s", boundary);
    size_t m = b + 4;
    memset(r, 0, sizeof(*r));
    if (n < b + 4 || memcmp(body, marker + 2, b + 2))
        goto malformed;
    size_t pos = b + 2;
    unsigned seen = 0;
    for (unsigned part = 0; part < 32; ++part) {
        if (pos + 2 > n || memcmp(body + pos, "\r\n", 2))
            goto malformed;
        pos += 2;
        const unsigned char *end = find(body + pos, n - pos, "\r\n\r\n", 4);
        if (!end || (size_t)(end - body) - pos > 2048)
            goto malformed;
        size_t hlen = (size_t)(end - body) - pos;
        char headers[2050], name[64] = {0};
        memcpy(headers, body + pos, hlen);
        headers[hlen] = 0;
        if (memchr(headers, 0, hlen))
            goto malformed;
        int disposition = 0, filename = 0;
        char *line = headers;
        while (*line) {
            char *next = strstr(line, "\r\n");
            if (next)
                *next = 0;
            for (const char *p = line; *p; ++p)
                if ((unsigned char)*p < 32 && *p != '\t')
                    goto malformed;
            char *colon = strchr(line, ':');
            if (!colon || colon == line)
                goto malformed;
            *colon++ = 0;
            while (*colon == ' ' || *colon == '\t')
                ++colon;
            if (!strcasecmp(line, "Content-Disposition")) {
                if (disposition++ || parameters(colon, "form-data", name, sizeof(name), &filename))
                    goto malformed;
            } else if (strcasecmp(line, "Content-Type"))
                goto malformed;
            if (!next)
                break;
            line = next + 2;
        }
        if (disposition != 1 || !name[0])
            goto malformed;
        const unsigned char *value = end + 4, *delimiter = value;
        for (;;) {
            delimiter = find(delimiter, n - (size_t)(delimiter - body), marker, m);
            if (!delimiter)
                goto malformed;
            size_t after = (size_t)(delimiter - body) + m;
            if (after + 2 <= n &&
                (!memcmp(body + after, "--", 2) || !memcmp(body + after, "\r\n", 2)))
                break;
            ++delimiter;
        }
        size_t len = (size_t)(delimiter - value);
        unsigned bit;
        if (!strcmp(name, "file"))
            bit = 1;
        else if (!strcmp(name, "model"))
            bit = 2;
        else if (!strcmp(name, "response_format"))
            bit = 4;
        else if (!strcmp(name, "language"))
            bit = 8;
        else if (!strcmp(name, "temperature"))
            bit = 16;
        else if (!strcmp(name, "prompt"))
            bit = 32;
        else if (!strcmp(name, "stream"))
            bit = 64;
        else if (!strcmp(name, "chunking_strategy"))
            bit = 128;
        else if (!strcmp(name, "known_speaker_names[]"))
            bit = 256;
        else if (!strcmp(name, "known_speaker_references[]"))
            bit = 512;
        else
            return wt_fail(e, 400, "Unsupported multipart parameter.", NULL,
                           "unsupported_parameter");
        if ((seen & bit) && bit < 256)
            return wt_fail(e, 400, "Duplicate multipart parameter.", NULL, "duplicate_parameter");
        seen |= bit;
        if (bit == 1) {
            if (!filename || !len)
                return wt_fail(e, 400, "A nonempty audio file is required.", "file",
                               "invalid_file");
            r->file = value;
            r->file_size = len;
        } else if (bit == 512) {
            if (filename || r->reference_count >= 4 || len > 430000 || !len ||
                memchr(value, 0, len))
                return wt_fail(e, 400, "Provide at most four 2–10 second WAV data URLs.",
                               "known_speaker_references", "invalid_value");
            r->references[r->reference_count] = value;
            r->reference_lengths[r->reference_count++] = len;
        } else {
            char text[257];
            if (filename || len >= sizeof(text) || memchr(value, 0, len))
                goto malformed;
            memcpy(text, value, len);
            text[len] = 0;
            if (bit == 256) {
                if (!len || len >= sizeof(r->names[0]) || r->name_count >= 4)
                    return wt_fail(e, 400,
                                   "Provide at most four nonempty names of at most 63 UTF-8 bytes.",
                                   "known_speaker_names", "invalid_value");
                for (size_t i = 0; i < len; ++i)
                    if ((unsigned char)text[i] < 32 || (unsigned char)text[i] == 127)
                        return wt_fail(e, 400, "Speaker names cannot contain control characters.",
                                       "known_speaker_names", "invalid_value");
                for (unsigned i = 0; i < r->name_count; ++i)
                    if (!strcmp(r->names[i], text))
                        return wt_fail(e, 400, "Speaker names must be unique.",
                                       "known_speaker_names", "invalid_value");
                memcpy(r->names[r->name_count++], text, len + 1);
            }
            if (bit == 2 && strcmp(text, "whisper-1") && strcmp(text, "whisper-large-v3-turbo") &&
                strcmp(text, "gpt-4o-transcribe-diarize"))
                return wt_fail(e, 400,
                               "Use whisper-1, whisper-large-v3-turbo, or "
                               "gpt-4o-transcribe-diarize (local aliases).",
                               "model", "model_not_found");
            if (bit == 4) {
                if (strcmp(text, "json") && strcmp(text, "text") && strcmp(text, "diarized_json"))
                    return wt_fail(e, 400, "Use json, text, or diarized_json.", "response_format",
                                   "unsupported_parameter");
                r->plain_text = !strcmp(text, "text");
                r->diarized_json = !strcmp(text, "diarized_json");
            }
            if (bit == 2)
                r->diarize = !strcmp(text, "gpt-4o-transcribe-diarize");
            if (bit == 64)
                r->stream = !strcmp(text, "true");
            if (bit == 128) {
                if (strcmp(text, "auto"))
                    return wt_fail(e, 400, "Only chunking_strategy=auto is supported.",
                                   "chunking_strategy", "unsupported_parameter");
                r->chunk_auto = 1;
            }
            if (bit == 8) {
                if (len < 2 || len > 3)
                    return wt_fail(e, 400, "Use a supported language code.", "language",
                                   "invalid_value");
                for (size_t i = 0; i < len; ++i)
                    if (text[i] < 'a' || text[i] > 'z')
                        return wt_fail(e, 400, "Use a lowercase language code.", "language",
                                       "invalid_value");
                memcpy(r->language, text, len + 1);
            }
            if (bit == 16) {
                char *tail;
                double temperature = strtod(text, &tail);
                if (!len || *tail || temperature != 0)
                    return wt_fail(e, 400, "Only temperature=0 greedy decoding is implemented.",
                                   "temperature", "unsupported_parameter");
            }
            if (bit == 32 && len)
                return wt_fail(e, 400, "Prompt conditioning is not implemented.", "prompt",
                               "unsupported_parameter");
            if (bit == 64 && strcmp(text, "false") && strcmp(text, "true"))
                return wt_fail(e, 400,
                               "stream must be true or false; Whisper responses are "
                               "non-streaming.",
                               "stream", "invalid_value");
        }
        pos = (size_t)(delimiter - body) + m;
        if (!memcmp(body + pos, "--", 2)) {
            pos += 2;
            if (pos + 2 == n && !memcmp(body + pos, "\r\n", 2))
                pos += 2;
            if (pos != n)
                goto malformed;
            if ((seen & 3) != 3)
                return wt_fail(e, 400, "file and model are required.",
                               (seen & 1) ? "model" : "file", "missing_required_parameter");
            if (r->name_count != r->reference_count)
                return wt_fail(e, 400, "Speaker names and references must have equal counts.",
                               "known_speaker_references", "invalid_value");
            if (!r->diarize && (r->diarized_json || r->chunk_auto || r->name_count))
                return wt_fail(e, 400, "Diarization options require gpt-4o-transcribe-diarize.",
                               "model", "invalid_value");
            if (!r->diarize)
                r->stream = 0;
            return 0;
        }
    }
malformed:
    return wt_fail(e, 400, "Malformed multipart/form-data body.", NULL, "invalid_multipart");
}
static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static unsigned u16(const unsigned char *p) { return p[0] | (unsigned)p[1] << 8; }
int wt_wav(const unsigned char *data, size_t n, const unsigned char **pcm, size_t *samples,
           wt_error *e) {
    int fmt = 0, found = 0;
    if (n < 12 || memcmp(data, "RIFF", 4) || memcmp(data + 8, "WAVE", 4) ||
        (uint64_t)u32(data + 4) + 8 != n)
        goto invalid;
    for (size_t p = 12; p < n;) {
        if (n - p < 8)
            goto invalid;
        uint32_t size = u32(data + p + 4);
        const unsigned char *chunk = data + p;
        p += 8;
        if (size > n - p || (size_t)size + (size & 1U) > n - p)
            goto invalid;
        if (!memcmp(chunk, "fmt ", 4)) {
            if (fmt++ || size < 16 || u16(data + p) != 1 || u16(data + p + 2) != 1 ||
                u32(data + p + 4) != 16000 || u32(data + p + 8) != 32000 ||
                u16(data + p + 12) != 2 || u16(data + p + 14) != 16)
                goto invalid;
        } else if (!memcmp(chunk, "data", 4)) {
            if (found++ || size < 2 || (size & 1))
                goto invalid;
            if (size / 2 > WT_AUDIO_LIMIT)
                return wt_fail(e, 413, "Audio exceeds the 300-second limit.", "file",
                               "audio_too_long");
            *pcm = data + p;
            *samples = size / 2;
        }
        p += size + (size_t)(size & 1U);
    }
    if (fmt == 1 && found == 1)
        return 0;
invalid:
    return wt_fail(e, 400, "Audio must be a valid mono PCM16 WAV at 16000 Hz.", "file",
                   "invalid_audio");
}
char *wt_json_string(const unsigned char *data, size_t n) {
    if (n > WT_TEXT_LIMIT)
        return NULL;
    char *out = malloc(n * 6 + 3);
    if (!out)
        return NULL;
    size_t j = 0;
    out[j++] = '"';
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n;) {
        unsigned c = data[i];
        if (c < 32 || c == '"' || c == '\\') {
            if (c >= 32) {
                out[j++] = '\\';
                out[j++] = (char)c;
            } else {
                memcpy(out + j, "\\u00", 4);
                j += 4;
                out[j++] = hex[c >> 4];
                out[j++] = hex[c & 15];
            }
            ++i;
        } else if (c < 128) {
            out[j++] = (char)c;
            ++i;
        } else {
            size_t count = c >= 0xc2 && c <= 0xdf   ? 2
                           : c >= 0xe0 && c <= 0xef ? 3
                           : c >= 0xf0 && c <= 0xf4 ? 4
                                                    : 0;
            int valid = count && count <= n - i;
            for (size_t k = 1; valid && k < count; ++k)
                valid = (data[i + k] & 0xc0) == 0x80;
            if (valid && count >= 3)
                valid = !(c == 0xe0 && data[i + 1] < 0xa0) && !(c == 0xed && data[i + 1] >= 0xa0) &&
                        !(c == 0xf0 && data[i + 1] < 0x90) && !(c == 0xf4 && data[i + 1] >= 0x90);
            if (valid) {
                memcpy(out + j, data + i, count);
                j += count;
                i += count;
            } else {
                memcpy(out + j, "\\ufffd", 6);
                j += 6;
                ++i;
            }
        }
    }
    out[j++] = '"';
    out[j] = 0;
    return out;
}
