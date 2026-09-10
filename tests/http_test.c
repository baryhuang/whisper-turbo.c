#define _POSIX_C_SOURCE 200809L
#include "api.h"
#include "languages.h"
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
static atomic_int stopping;
static unsigned port;
static unsigned calls;
static unsigned char wave[364];
static void le32(unsigned char *p, unsigned n) {
    p[0] = n;
    p[1] = n >> 8;
    p[2] = n >> 16;
    p[3] = n >> 24;
}
static void make_wave(void) {
    memcpy(wave, "RIFF", 4);
    le32(wave + 4, sizeof(wave) - 8);
    memcpy(wave + 8, "WAVEfmt ", 8);
    le32(wave + 16, 16);
    wave[20] = 1;
    wave[22] = 1;
    le32(wave + 24, 16000);
    le32(wave + 28, 32000);
    wave[32] = 2;
    wave[34] = 16;
    memcpy(wave + 36, "data", 4);
    le32(wave + 40, sizeof(wave) - 44);
}
static size_t multipart(unsigned char *out, const char *fields) {
    size_t n = (size_t)sprintf(
        (char *)out,
        "--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n%s"
        "--test\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"../../private.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n",
        fields);
    memcpy(out + n, wave, sizeof(wave));
    n += sizeof(wave);
    memcpy(out + n, "\r\n--test--\r\n", 12);
    return n + 12;
}
static int backend(void *unused, const wt_request *r, wt_result *out, wt_error *e, wt_cancel cancel,
                   void *context) {
    (void)unused;
    ++calls; /* Only the persistent inference thread touches this counter. */
    if (!strcmp(r->language, "zz")) {
        struct timespec pause = {0, 10000000};
        for (int i = 0; i < 300; ++i) {
            if (cancel(context))
                return wt_fail(e, 504, "Cancelled.", NULL, "request_timeout");
            nanosleep(&pause, NULL);
        }
    }
    if (!strcmp(r->language, "xx"))
        return wt_fail(e, 500, "Synthetic failure.", NULL, "inference_error");
    const char text[] = "Hello \"C\"\n\xc3\xa9";
    out->text = malloc(sizeof(text));
    CHECK(out->text);
    memcpy(out->text, text, sizeof(text));
    out->length = sizeof(text) - 1;
    if (r->diarize) {
        out->duration = 0.01;
        out->segments = calloc(1, sizeof(wt_segment));
        CHECK(out->segments);
        out->segment_count = 1;
        out->segments[0] = (wt_segment){.start = 0, .end = 0.01, .speaker = "A",
                                       .text = malloc(sizeof(text)), .length = sizeof(text)-1};
        CHECK(out->segments[0].text);
        memcpy(out->segments[0].text, text, sizeof(text));
    }
    return 0;
}
static void *serve(void *unused) {
    (void)unused;
    wt_server_options options = {.host = "127.0.0.1",
                                 .port = port,
                                 .api_key = "test-secret",
                                 .timeout_seconds = 1,
                                 .upload_seconds = 1};
    CHECK(!wt_serve(&options, backend, NULL, &stopping));
    return NULL;
}
static int connect_server(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    struct sockaddr_in address = {.sin_family = AF_INET,
                                  .sin_port = htons((uint16_t)port),
                                  .sin_addr.s_addr = htonl(0x7f000001U)};
    if (connect(fd, (struct sockaddr *)&address, sizeof(address))) {
        close(fd);
        return -1;
    }
    struct timeval timeout = {5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    return fd;
}
static void send_bytes(int fd, const void *data, size_t n) {
    const unsigned char *p = data;
    while (n) {
        ssize_t k = send(fd, p, n, 0);
        CHECK(k > 0);
        p += k;
        n -= (size_t)k;
    }
}
static int receive(int fd, char *out, size_t capacity) {
    size_t n = 0;
    while (n + 1 < capacity) {
        ssize_t k = recv(fd, out + n, capacity - n - 1, 0);
        if (k == 0 || (k < 0 && errno == ECONNRESET))
            break;
        CHECK(k > 0);
        n += (size_t)k;
    }
    out[n] = 0;
    int status = 0;
    CHECK(sscanf(out, "HTTP/1.1 %d", &status) == 1);
    close(fd);
    return status;
}
static int request(const char *headers, const void *body, size_t n, char *response) {
    int fd = connect_server();
    CHECK(fd >= 0);
    send_bytes(fd, headers, strlen(headers));
    if (n)
        send_bytes(fd, body, n);
    return receive(fd, response, 4096);
}
static void post_header(char *header, size_t n, const char *extra) {
    sprintf(header,
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\n"
            "Authorization: Bearer test-secret\r\nContent-Type: multipart/form-data; "
            "boundary=\"test\"\r\n"
            "Content-Length: %zu\r\n%s\r\n",
            n, extra);
}
static void unit(void) {
    CHECK(!strcmp(wt_languages[0], "en") && !strcmp(wt_languages[1], "zh") &&
          !strcmp(wt_languages[99], "yue"));
    char boundary[71];
    CHECK(!wt_boundary("multipart/form-data; boundary=\"x:y\"", boundary));
    CHECK(!strcmp(boundary, "x:y"));
    CHECK(wt_boundary("multipart/form-data; boundary=a; boundary=b", boundary));
    CHECK(wt_boundary("multipart/form-data; boundary=\"bad\r\nvalue\"", boundary));
    CHECK(wt_boundary("multipart/form-data; boundary=\"unterminated", boundary));
    unsigned char body[4096];
    wt_request r;
    wt_error e;
    size_t n = multipart(body, "");
    CHECK(!wt_multipart(body, n, "test", &r, &e));
    CHECK(r.file_size == sizeof(wave) && !memcmp(r.file, wave, sizeof(wave)));
    /* A boundary prefix inside binary audio is not a MIME delimiter. */
    memcpy((unsigned char *)r.file + 80, "\r\n--testNOT_A_BOUNDARY", 22);
    CHECK(!wt_multipart(body, n, "test", &r, &e));
    for (size_t truncated = 0; truncated < n - 2; ++truncated)
        CHECK(wt_multipart(body, truncated, "test", &r, &e));
    n = multipart(body,
                  "--test\r\nContent-Disposition: form-data; name=model\r\n\r\nwhisper-1\r\n");
    CHECK(wt_multipart(body, n, "test", &r, &e) && !strcmp(e.code, "duplicate_parameter"));
    n = multipart(
        body,
        "--test\r\nContent-Disposition: form-data; name=response_format\r\n\r\nverbose_json\r\n");
    CHECK(wt_multipart(body, n, "test", &r, &e) && !strcmp(e.param, "response_format"));
    n = multipart(body,
                  "--test\r\nContent-Disposition: form-data; name=temperature\r\n\r\nnan\r\n");
    CHECK(wt_multipart(body, n, "test", &r, &e));
    n = multipart(body, "--test\r\nContent-Disposition: form-data; name=prompt\r\n\r\nhello\r\n");
    CHECK(wt_multipart(body, n, "test", &r, &e) && !strcmp(e.param, "prompt"));
    const unsigned char *pcm;
    size_t samples;
    CHECK(!wt_wav(wave, sizeof(wave), &pcm, &samples, &e) && samples == 160);
    for (size_t i = 0; i < sizeof(wave); ++i)
        CHECK(wt_wav(wave, i, &pcm, &samples, &e));
    unsigned char copy[sizeof(wave)];
    memcpy(copy, wave, sizeof(copy));
    copy[22] = 2;
    CHECK(wt_wav(copy, sizeof(copy), &pcm, &samples, &e));
    memcpy(copy, wave, sizeof(copy));
    le32(copy + 40, 0xffffffffU);
    CHECK(wt_wav(copy, sizeof(copy), &pcm, &samples, &e));
    char *json = wt_json_string((const unsigned char *)"\"\n\xc3\xa9\xff", 5);
    CHECK(json && !strcmp(json, "\"\\\"\\u000a\xc3\xa9\\ufffd\""));
    free(json);
    /* Deterministic malformed-input smoke fuzz, including embedded NULs. */
    uint32_t random = 7;
    for (int i = 0; i < 10000; ++i) {
        random = random * 1664525U + 1013904223U;
        size_t length = random % sizeof(body);
        for (size_t j = 0; j < length; ++j) {
            random = random * 1664525U + 1013904223U;
            body[j] = random >> 24;
        }
        (void)wt_multipart(body, length, "test", &r, &e);
        (void)wt_wav(body, length, &pcm, &samples, &e);
    }
}
int main(void) {
    signal(SIGPIPE, SIG_IGN);
    make_wave();
    unit();
    int probe = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(probe >= 0);
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(0x7f000001U)};
    CHECK(!bind(probe, (struct sockaddr *)&address, sizeof(address)));
    socklen_t size = sizeof(address);
    CHECK(!getsockname(probe, (struct sockaddr *)&address, &size));
    port = ntohs(address.sin_port);
    close(probe);
    pthread_t thread;
    CHECK(!pthread_create(&thread, NULL, serve, NULL));
    struct timespec pause = {0, 10000000};
    int ready = -1;
    for (int i = 0; i < 200 && ready < 0; ++i) {
        nanosleep(&pause, NULL);
        ready = connect_server();
    }
    CHECK(ready >= 0);
    close(ready);
    char response[4096], header[2048];
    unsigned char body[4096];
    CHECK(request("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", NULL, 0, response) == 200);
    CHECK(strstr(response, "{\"status\":\"ready\"}"));
    CHECK(request("POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                  "0\r\n\r\n",
                  NULL, 0, response) == 401);
    CHECK(strstr(response, "\"type\":\"authentication_error\""));
    size_t n = multipart(body, "");
    post_header(header, n, "");
    CHECK(request(header, body, n, response) == 200);
    CHECK(strstr(response, "{\"text\":\"Hello \\\"C\\\"\\u000a\xc3\xa9\"}"));
    CHECK(!strstr(response, "Synthetic") && strstr(response, "Content-Type: application/json"));
    for (int i = 0; i < 10; ++i)
        CHECK(request(header, body, n, response) == 200);
    n = multipart(body,
                  "--test\r\nContent-Disposition: form-data; name=response_format\r\n\r\ntext\r\n");
    post_header(header, n, "");
    CHECK(request(header, body, n, response) == 200);
    CHECK(strstr(response, "Content-Type: text/plain") &&
          strstr(response, "Hello \"C\"\n\xc3\xa9"));
    post_header(header, n, "Expect: 100-continue\r\n");
    int fd = connect_server();
    CHECK(fd >= 0);
    send_bytes(fd, header, strlen(header));
    char interim[26] = {0};
    size_t got = 0;
    while (got < 25) {
        ssize_t k = recv(fd, interim + got, 25 - got, 0);
        CHECK(k > 0);
        got += (size_t)k;
    }
    CHECK(!strcmp(interim, "HTTP/1.1 100 Continue\r\n\r\n"));
    send_bytes(fd, body, n);
    CHECK(receive(fd, response, sizeof(response)) == 200);
    post_header(header, n, "Content-Length: 1\r\n");
    CHECK(request(header, NULL, 0, response) == 400);
    post_header(header, n, "Transfer-Encoding: chunked\r\n");
    CHECK(request(header, NULL, 0, response) == 400);
    post_header(header, WT_UPLOAD_LIMIT + 1U, "");
    CHECK(request(header, NULL, 0, response) == 413);
    post_header(header, n, "");
    fd = connect_server();
    CHECK(fd >= 0);
    send_bytes(fd, header, strlen(header));
    CHECK(receive(fd, response, sizeof(response)) == 408);
    n = multipart(body, "--test\r\nContent-Disposition: form-data; name=language\r\n\r\nzz\r\n");
    post_header(header, n, "");
    fd = connect_server();
    CHECK(fd >= 0);
    send_bytes(fd, header, strlen(header));
    send_bytes(fd, body, n);
    struct timespec settle = {0, 100000000};
    nanosleep(&settle, NULL);
    CHECK(request(header, NULL, 0, response) == 429);
    CHECK(request("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", NULL, 0, response) == 200);
    CHECK(receive(fd, response, sizeof(response)) == 504);
    /* A disconnected inference is cancelled, then the worker accepts another request. */
    fd = connect_server();
    CHECK(fd >= 0);
    send_bytes(fd, header, strlen(header));
    send_bytes(fd, body, n);
    nanosleep(&settle, NULL);
    close(fd);
    nanosleep(&settle, NULL);
    n = multipart(body, "--test\r\nContent-Disposition: form-data; name=language\r\n\r\nxx\r\n");
    post_header(header, n, "");
    CHECK(request(header, body, n, response) == 500);
    n = multipart(body, "");
    post_header(header, n, "");
    CHECK(request(header, body, n, response) == 200);
    int idle[8];
    nanosleep(&settle, NULL);
    for (int i = 0; i < 8; ++i) {
        idle[i] = connect_server();
        CHECK(idle[i] >= 0);
    }
    nanosleep(&settle, NULL);
    CHECK(request("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", NULL, 0, response) == 503);
    CHECK(strstr(response, "\"code\":\"server_busy\""));
    for (int i = 0; i < 8; ++i)
        close(idle[i]);
    nanosleep(&settle, NULL);
    CHECK(request("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", NULL, 0, response) == 200);
    for (int stream = 0; stream < 2; ++stream) {
        n = multipart(body, stream ? "--test\r\nContent-Disposition: form-data; name=response_format\r\n\r\ndiarized_json\r\n--test\r\nContent-Disposition: form-data; name=stream\r\n\r\ntrue\r\n"
                                   : "--test\r\nContent-Disposition: form-data; name=response_format\r\n\r\ndiarized_json\r\n");
        char *model = strstr((char *)body, "whisper-1");
        CHECK(model);
        size_t off = (size_t)(model-(char *)body), extra = strlen("gpt-4o-transcribe-diarize")-9;
        memmove(body+off+9+extra, body+off+9, n-off-9);
        memcpy(body+off, "gpt-4o-transcribe-diarize", 9+extra); n += extra;
        post_header(header,n,"");
        CHECK(request(header,body,n,response)==200);
        CHECK(strstr(response,"\"speaker\":\"A\"") && strstr(response,"\"id\":\"seg_001\""));
        if (stream) CHECK(strstr(response,"Content-Type: text/event-stream") && strstr(response,"transcript.text.done"));
        else CHECK(strstr(response,"\"usage\":{\"type\":\"duration\""));
    }
    stopping = 1;
    CHECK(!pthread_join(thread, NULL));
    CHECK(calls >= 16);
    puts("C HTTP tests passed: multipart, auth, JSON/text, bounds, overload, timeout, "
         "cancellation, recovery.");
    return 0;
}
