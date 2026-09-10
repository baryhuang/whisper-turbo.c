#include "api.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void field(char *b, const char *name, const char *v) {
    snprintf(b + strlen(b), 4096 - strlen(b), "--x\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n", name,
            v);
}
static int parse(char *b, wt_request *r, wt_error *e) {
    strcat(b, "--x\r\nContent-Disposition: form-data; name=\"file\"; "
              "filename=\"a.wav\"\r\n\r\nx\r\n--x--\r\n");
    return wt_multipart((unsigned char *)b, strlen(b), "x", r, e);
}
static void le(unsigned char *p, unsigned v) {
    for (int i = 0; i < 4; ++i)
        p[i] = (unsigned char)(v >> (i * 8));
}
int main(void) {
    char b[4096] = {0};
    wt_request req;
    wt_error e = {0};
    field(b, "response_format", "diarized_json");
    field(b, "stream", "true");
    field(b, "chunking_strategy", "auto");
    field(b, "model", "gpt-4o-transcribe-diarize");
    assert(!parse(b, &req, &e) && req.diarize && req.diarized_json && req.stream && req.chunk_auto);
    b[0] = 0;
    field(b, "model", "whisper-1");
    field(b, "response_format", "diarized_json");
    assert(parse(b, &req, &e));
    b[0] = 0;
    field(b, "model", "gpt-4o-transcribe-diarize");
    field(b, "chunking_strategy", "manual");
    assert(parse(b, &req, &e));
    b[0] = 0;
    field(b, "model", "gpt-4o-transcribe-diarize");
    field(b, "known_speaker_names[]", "agent");
    assert(parse(b, &req, &e));
    b[0] = 0;
    field(b, "model", "gpt-4o-transcribe-diarize");
    field(b, "known_speaker_names[]", "agent");
    field(b, "known_speaker_references[]", "data:audio/wav;base64,AAAA");
    assert(!parse(b, &req, &e) && req.name_count == 1);
    wt_segment s[3] = {
        {.start = 0, .end = 1, .speaker = "A", .text = (unsigned char *)"Hello \"C\"", .length = 9},
        {.start = 2, .end = 3, .speaker = "agent", .text = (unsigned char *)" Hi", .length = 3},
        {.start = 4, .end = 5, .speaker = "A", .text = (unsigned char *)" Again", .length = 6}};
    wt_result result = {.text = (unsigned char *)"Hello \"C\" Hi Again",
                        .length = 18,
                        .duration = 6,
                        .segments = s,
                        .segment_count = 3};
    req = (wt_request){.diarize = 1, .diarized_json = 1};
    char *out = NULL;
    size_t n = 0;
    const char *type = NULL;
    assert(!wt_render(&req, &result, &out, &n, &type));
    assert(!strcmp(type, "application/json") && n == strlen(out));
    assert(strstr(out, "\"task\":\"transcribe\"") && strstr(out, "\"id\":\"seg_003\"") &&
           strstr(out, "\"type\":\"duration\""));
    assert(strstr(out, "Hello \\\"C\\\""));
    free(out);
    req.stream = 1;
    assert(!wt_render(&req, &result, &out, &n, &type));
    assert(strstr(type, "text/event-stream") && strstr(out, "\"segment_id\":\"seg_001\""));
    assert(strstr(out, "\"type\":\"transcript.text.done\"") && !strstr(out, "\"usage\""));
    /* Joining deltas preserves the spaces in the complete transcript. */
    assert(strstr(out, "\"delta\":\"Hello \\\"C\\\"\"") &&
           strstr(out, "\"delta\":\" Hi\"") && strstr(out, "\"delta\":\" Again\""));
    free(out);
    req.stream = 0;
    req.diarized_json = 0;
    assert(!wt_render(&req, &result, &out, &n, &type) && !strstr(out, "\"segments\""));
    free(out);
    req.plain_text = 1;
    assert(!wt_render(&req, &result, &out, &n, &type) && n == result.length);
    free(out);
    s[1].start = NAN;
    assert(wt_render(&req, &result, &out, &n, &type));
    s[1].start = 0.5;
    assert(wt_render(&req, &result, &out, &n, &type));
    s[1].start = 2;
    s[2].end = 7;
    assert(wt_render(&req, &result, &out, &n, &type));
    s[2].end = 5;
    result.segment_count = WT_SEGMENT_LIMIT + 1;
    assert(wt_render(&req, &result, &out, &n, &type));
    unsigned char *pcm = NULL;
    size_t samples = 0;
    assert(wt_reference_wav((unsigned char *)"data:audio/wav;base64,AA=A", 25, &pcm, &samples, &e));
    size_t bytes = 64044;
    unsigned char *wav = calloc(bytes, 1);
    assert(wav);
    memcpy(wav, "RIFF", 4);
    le(wav + 4, (unsigned)bytes - 8);
    memcpy(wav + 8, "WAVEfmt ", 8);
    le(wav + 16, 16);
    wav[20] = 1;
    wav[22] = 1;
    le(wav + 24, 16000);
    le(wav + 28, 32000);
    wav[32] = 2;
    wav[34] = 16;
    memcpy(wav + 36, "data", 4);
    le(wav + 40, 64000);
    const char abc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *url = calloc(bytes * 2, 1);
    assert(url);
    strcpy(url, "data:audio/wav;base64,");
    size_t pos = strlen(url);
    for (size_t i = 0; i < bytes; i += 3) {
        unsigned v = (unsigned)wav[i] << 16;
        if (i + 1 < bytes)
            v |= (unsigned)wav[i + 1] << 8;
        if (i + 2 < bytes)
            v |= wav[i + 2];
        url[pos++] = abc[v >> 18];
        url[pos++] = abc[(v >> 12) & 63];
        url[pos++] = i + 1 < bytes ? abc[(v >> 6) & 63] : '=';
        url[pos++] = i + 2 < bytes ? abc[v & 63] : '=';
    }
    assert(!wt_reference_wav((unsigned char *)url, pos, &pcm, &samples, &e) && samples == 32000);
    free(pcm);
    url[pos - 1] = '!';
    assert(wt_reference_wav((unsigned char *)url, pos, &pcm, &samples, &e));
    free(wav);
    free(url);
    puts("C diarized API tests passed: multipart options, references, "
         "JSON/text/SSE schemas, bounds.");
    return 0;
}
