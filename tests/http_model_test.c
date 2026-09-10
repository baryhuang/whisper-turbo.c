#define _POSIX_C_SOURCE 200809L
#include "api.h"
#include <arpa/inet.h>
#include <errno.h>
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
static unsigned port;
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int connect_api(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_port = htons((uint16_t)port),
                            .sin_addr.s_addr = htonl(0x7f000001U)};
    if (connect(fd, (struct sockaddr *)&a, sizeof(a))) {
        close(fd);
        return -1;
    }
    struct timeval timeout = {600, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    return fd;
}
static void send_bytes(int fd, const void *data, size_t n) {
    const unsigned char *p = data;
    while (n) {
        ssize_t k = send(fd, p, n, 0);
        CHECK(k > 0);
        n -= (size_t)k;
        p += k;
    }
}
static char *response(int fd, int expected) {
    char *buffer = calloc(1, WT_TEXT_LIMIT * 6 + 8192);
    CHECK(buffer);
    size_t n = 0, capacity = WT_TEXT_LIMIT * 6 + 8192;
    for (;;) {
        CHECK(n + 1 < capacity);
        ssize_t k = recv(fd, buffer + n, capacity - n - 1, 0);
        if (!k)
            break;
        CHECK(k > 0);
        n += (size_t)k;
    }
    close(fd);
    int status = 0;
    CHECK(sscanf(buffer, "HTTP/1.1 %d", &status) == 1);
    if (status != expected) {
        fprintf(stderr, "Unexpected HTTP status %d (expected %d)\n", status, expected);
        if (status >= 400)
            fprintf(stderr, "%s\n", buffer);
    }
    CHECK(status == expected);
    char *body = strstr(buffer, "\r\n\r\n");
    CHECK(body);
    body += 4;
    char *length = strstr(buffer, "Content-Length:");
    CHECK(length);
    CHECK(strtoul(length + 15, NULL, 10) == n - (size_t)(body - buffer));
    memmove(buffer, body, n - (size_t)(body - buffer) + 1);
    return buffer;
}
static int begin_post(const unsigned char *wave, size_t bytes, int text, int autodetect) {
    char prefix[1024];
    int size = snprintf(
        prefix, sizeof(prefix),
        "--bench\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n"
        "%s--bench\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n%s\r\n"
        "--bench\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"fixture.wav\"\r\nContent-Type: audio/wav\r\n\r\n",
        autodetect ? ""
                   : "--bench\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n",
        text ? "text" : "json");
    const char suffix[] = "\r\n--bench--\r\n";
    char header[512];
    int h = snprintf(header, sizeof(header),
                     "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: "
                     "multipart/form-data; boundary=bench\r\nContent-Length: %zu\r\n\r\n",
                     (size_t)size + bytes + sizeof(suffix) - 1);
    int fd = connect_api();
    CHECK(fd >= 0);
    send_bytes(fd, header, (size_t)h);
    send_bytes(fd, prefix, (size_t)size);
    send_bytes(fd, wave, bytes);
    send_bytes(fd, suffix, sizeof(suffix) - 1);
    return fd;
}
static void health(void) {
    int fd = connect_api();
    CHECK(fd >= 0);
    const char get[] = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    send_bytes(fd, get, sizeof(get) - 1);
    char *r = response(fd, 200);
    CHECK(!strcmp(r, "{\"status\":\"ready\"}"));
    free(r);
}
static void le32(unsigned char *p, size_t value) {
    for (unsigned i = 0; i < 4; ++i)
        p[i] = (unsigned char)(value >> (8 * i));
}
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s PORT JFK.wav [smoke|repeat|limits]\n", argv[0]);
        return 2;
    }
    port = (unsigned)strtoul(argv[1], NULL, 10);
    CHECK(port && port <= 65535);
    struct timespec pause = {0, 100000000};
    int ready = -1;
    for (int i = 0; i < 100 && ready < 0; ++i) {
        nanosleep(&pause, NULL);
        ready = connect_api();
    }
    CHECK(ready >= 0);
    close(ready);
    health();
    FILE *f = fopen(argv[2], "rb");
    CHECK(f);
    CHECK(!fseek(f, 0, SEEK_END));
    long size = ftell(f);
    CHECK(size > 0 && size <= WT_UPLOAD_LIMIT);
    rewind(f);
    unsigned char *wav = malloc((size_t)size);
    CHECK(wav);
    CHECK(fread(wav, 1, (size_t)size, f) == (size_t)size);
    fclose(f);
    const unsigned char *pcm;
    size_t samples;
    wt_error error;
    CHECK(!wt_wav(wav, (size_t)size, &pcm, &samples, &error));
    int repeat = argc == 4 && !strcmp(argv[3], "repeat");
    int limits = argc == 4 && !strcmp(argv[3], "limits");
    CHECK(argc == 3 || repeat || limits || !strcmp(argv[3], "smoke"));
    char *first = NULL;
    for (int trial = 0; trial < (repeat ? 10 : argc == 4 ? 1 : 3); ++trial) {
        double started = now();
        int fd = begin_post(wav, (size_t)size, 0, trial == 2);
        if (!trial && argc == 3) {
            nanosleep(&pause, NULL);
            health();
            const char busy[] =
                "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: "
                "multipart/form-data; boundary=bench\r\nContent-Length: 10\r\n\r\n";
            int other = connect_api();
            CHECK(other >= 0);
            send_bytes(other, busy, sizeof(busy) - 1);
            char *r = response(other, 429);
            free(r);
        }
        char *r = response(fd, 200);
        CHECK(strstr(r, "\"text\":\"") && strstr(r, "country") && strlen(r) > 50);
        if (!trial)
            first = strdup(r);
        else
            CHECK(!strcmp(r, first));
        printf("HTTP_RESULT "
               "{\"case\":\"jfk_%s\",\"trial\":%d,\"audio_seconds\":%.3f,\"seconds\":%.6f,"
               "\"status\":200,\"transcript_checked\":true}\n",
               trial == 2 ? "autodetect" : "english", trial + 1, samples / 16000.0,
               now() - started);
        free(r);
    }
    if (argc == 3) {
        /* 31 seconds: exact digital silence followed by speech entirely after 30s.
         * A first-window-only implementation would incorrectly return empty text. */
        size_t long_samples = 31 * 16000U + samples;
        size_t bytes = 44 + 2 * long_samples;
        unsigned char *long_wav = calloc(1, bytes);
        CHECK(long_wav);
        memcpy(long_wav, "RIFF", 4);
        le32(long_wav + 4, bytes - 8);
        memcpy(long_wav + 8, "WAVEfmt ", 8);
        le32(long_wav + 16, 16);
        long_wav[20] = 1;
        long_wav[22] = 1;
        le32(long_wav + 24, 16000);
        le32(long_wav + 28, 32000);
        long_wav[32] = 2;
        long_wav[34] = 16;
        memcpy(long_wav + 36, "data", 4);
        le32(long_wav + 40, bytes - 44);
        memcpy(long_wav + 44 + 31 * 32000U, pcm, samples * 2);
        double started = now();
        char *r = response(begin_post(long_wav, bytes, 1, 0), 200);
        CHECK(strstr(r, "country") && strlen(r) > 50);
        printf("HTTP_RESULT "
               "{\"case\":\"speech_after_30s_text\",\"audio_seconds\":%.3f,\"seconds\":%.6f,"
               "\"status\":200,\"transcript_checked\":true}\n",
               long_samples / 16000.0, now() - started);
        free(r);
        free(long_wav);
    }
    if (limits) {
        for (int test = 0; test < 2; ++test) {
            size_t count = test ? samples : WT_AUDIO_LIMIT;
            size_t bytes = test ? 24999000U : 44 + count * 2;
            unsigned char *input = calloc(1, bytes);
            CHECK(input);
            memcpy(input, "RIFF", 4);
            le32(input + 4, bytes - 8);
            memcpy(input + 8, "WAVEfmt ", 8);
            le32(input + 16, 16);
            input[20] = 1;
            input[22] = 1;
            le32(input + 24, 16000);
            le32(input + 28, 32000);
            input[32] = 2;
            input[34] = 16;
            size_t data_start = 36;
            if (test) {
                size_t junk = bytes - 52 - count * 2;
                memcpy(input + 36, "JUNK", 4);
                le32(input + 40, junk);
                data_start += 8 + junk;
            }
            memcpy(input + data_start, "data", 4);
            le32(input + data_start + 4, count * 2);
            for (size_t i = 0; i < count; ++i)
                memcpy(input + data_start + 8 + i * 2, pcm + (i % samples) * 2, 2);
            double started = now();
            char *r = response(begin_post(input, bytes, 1, 0), 200);
            size_t mentions = 0;
            for (const char *p = r; (p = strstr(p, "country")) != NULL; p += 7)
                ++mentions;
            printf("HTTP_CONTENT {\"case\":\"%s\",\"text_bytes\":%zu,\"country_mentions\":%zu}\n",
                   test ? "near_upload_limit" : "120s_repeated_speech", strlen(r), mentions);
            /* Each original JFK clip says 'country' twice. Require more content
             * than its first 30 seconds, not an arbitrary character count. */
            CHECK(mentions >= (test ? 2U : 8U) && strlen(r) > 50);
            printf("HTTP_RESULT {\"case\":\"%s\",\"audio_seconds\":%.3f,\"file_bytes\":%zu,"
                   "\"seconds\":%.6f,\"status\":200,\"transcript_checked\":true}\n",
                   test ? "near_upload_limit" : "120s_repeated_speech", count / 16000.0, bytes,
                   now() - started);
            free(r);
            free(input);
        }
    }
    free(first);
    free(wav);
    health();
    puts("Real-model C HTTP checks passed; no transcript content logged.");
    return 0;
}
