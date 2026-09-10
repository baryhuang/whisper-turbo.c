#define _POSIX_C_SOURCE 200809L
#include "api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void wt_result_free(wt_result *r) {
    for (size_t i = 0; i < r->segment_count; ++i)
        free(r->segments[i].text);
    free(r->segments);
    free(r->words);
    free(r->text);
    memset(r, 0, sizeof(*r));
}
static int segment(FILE *f, const wt_segment *s, size_t id) {
    char *speaker = wt_json_string((const unsigned char *)s->speaker, strlen(s->speaker));
    char *text = wt_json_string(s->text, s->length);
    if (!speaker || !text) {
        free(speaker);
        free(text);
        return -1;
    }
    int rc = fprintf(f,
                     "{\"type\":\"transcript.text.segment\",\"id\":\"seg_%03zu\","
                     "\"start\":%.6f,\"end\":%.6f,\"speaker\":%s,\"text\":%s}",
                     id + 1, s->start, s->end, speaker, text);
    free(speaker);
    free(text);
    return rc < 0 ? -1 : 0;
}
int wt_render(const wt_request *req, const wt_result *r, char **out, size_t *length,
              const char **type) {
    *out = NULL;
    *length = 0;
    if (!r->text || r->length > WT_TEXT_LIMIT || r->segment_count > WT_SEGMENT_LIMIT ||
        (r->segment_count && !r->segments) || !isfinite(r->duration) || r->duration < 0 ||
        r->duration > 120)
        return -1;
    size_t total = 0;
    double previous = 0;
    for (size_t i = 0; i < r->segment_count; ++i) {
        const wt_segment *s = r->segments + i;
        if (!isfinite(s->start) || !isfinite(s->end) || s->start < previous || s->end <= s->start ||
            s->end > r->duration || !s->text || s->length > WT_TEXT_LIMIT - total ||
            !memchr(s->speaker, 0, sizeof(s->speaker)) || !s->speaker[0])
            return -1;
        total += s->length;
        previous = s->end;
    }
    FILE *f = open_memstream(out, length);
    if (!f)
        return -1;
    char *quoted = wt_json_string(r->text, r->length);
    int failed = !quoted;
    if (failed)
        goto done;
    if (req->stream) {
        *type = "text/event-stream; charset=utf-8";
        if (req->diarized_json) {
            for (size_t i = 0; i < r->segment_count; ++i) {
                char *q = wt_json_string(r->segments[i].text, r->segments[i].length);
                if (!q) {
                    failed = 1;
                    break;
                }
                fprintf(f,
                        "data: "
                        "{\"type\":\"transcript.text.delta\",\"segment_id\":\"seg_%"
                        "03zu\",\"delta\":%s}\n\n",
                        i + 1, q);
                free(q);
                fputs("data: ", f);
                if (segment(f, r->segments + i, i)) {
                    failed = 1;
                    break;
                }
                fputs("\n\n", f);
            }
        } else
            fprintf(f, "data: {\"type\":\"transcript.text.delta\",\"delta\":%s}\n\n", quoted);
        fprintf(f, "data: {\"type\":\"transcript.text.done\",\"text\":%s}\n\n", quoted);
    } else if (req->plain_text) {
        *type = "text/plain; charset=utf-8";
        if (fwrite(r->text, 1, r->length, f) != r->length)
            failed = 1;
    } else {
        *type = "application/json";
        if (req->diarized_json) {
            fprintf(f,
                    "{\"task\":\"transcribe\",\"duration\":%.6f,\"text\":%s,"
                    "\"segments\":[",
                    r->duration, quoted);
            for (size_t i = 0; i < r->segment_count; ++i) {
                if (i)
                    fputc(',', f);
                if (segment(f, r->segments + i, i)) {
                    failed = 1;
                    break;
                }
            }
            fputc(']', f);
        } else
            fprintf(f, "{\"text\":%s", quoted);
        if (req->diarize)
            fprintf(f, ",\"usage\":{\"type\":\"duration\",\"seconds\":%.6f}", r->duration);
        fputc('}', f);
    }
done:
    free(quoted);
    failed |= ferror(f) != 0;
    failed |= fclose(f) != 0;
    if (*length > WT_TEXT_LIMIT * 20U + WT_SEGMENT_LIMIT * 1024U)
        failed = 1;
    if (failed) {
        free(*out);
        *out = NULL;
        *length = 0;
        return -1;
    }
    return 0;
}
