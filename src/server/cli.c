#define _POSIX_C_SOURCE 200809L
#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s MODEL.whtrbo DIARIZATION_DIRECTORY AUDIO.wav [LANGUAGE]\n", argv[0]);
        return 2;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    wt_engine engine = {.threads = 8, .diarization_directory = argv[2]};
    const char *threads = getenv("OMP_NUM_THREADS");
    if (threads) {
        char *tail;
        long n = strtol(threads, &tail, 10);
        if (!*threads || *tail || n < 1 || n > 8) return 2;
        engine.threads = (unsigned)n;
    }
    wt_request req = {.diarize = 1, .diarized_json = 1, .chunk_auto = 1};
    if (argc == 5) {
        if (strlen(argv[4]) < 2 || strlen(argv[4]) > 3) return 2;
        strcpy(req.language, argv[4]);
    }
    FILE *f = fopen(argv[3], "rb");
    if (!f) return 2;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return 2; }
    long bytes = ftell(f);
    if (bytes <= 0 || bytes > WT_UPLOAD_LIMIT || fseek(f, 0, SEEK_SET)) { fclose(f); return 2; }
    unsigned char *data = malloc((size_t)bytes);
    if (!data) { fclose(f); return 2; }
    int bad = fread(data, 1, (size_t)bytes, f) != (size_t)bytes;
    fclose(f);
    if (bad) { free(data); return 2; }
    req.file = data; req.file_size = (size_t)bytes;
    engine.model.image.fd = -1;
    if (cllm_whisper_turbo_model_open(argv[1], &engine.model)) { free(data); return 2; }
#ifdef WT_RESIDENT_BENCH
    char *expected = NULL;
    size_t expected_length = 0;
    for (unsigned trial = 0; trial < 4; ++trial) {
        wt_result r = {0}; wt_error e = {0};
        char *body = NULL; size_t length = 0; const char *type = NULL;
        clock_gettime(CLOCK_MONOTONIC, &start);
        int rc = wt_transcribe(&engine, &req, &r, &e, NULL, NULL);
        if (!rc) rc = wt_render(&req, &r, &body, &length, &type);
        clock_gettime(CLOCK_MONOTONIC, &end);
        if (!rc && expected && (length != expected_length || memcmp(body, expected, length))) rc = -1;
        fprintf(stderr, "RESIDENT {\"trial\":%u,\"warmup\":%s,\"asr_windows\":%zu,\"diarization_passes\":%zu,\"seconds\":%.6f,\"success\":%s}\n",
                trial, trial ? "false" : "true", r.asr_windows, r.diarization_passes,
                end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9, rc ? "false" : "true");
        wt_result_free(&r);
        if (rc) { fprintf(stderr,"%s\n",e.message ? e.message : "Result mismatch"); free(body); free(expected); free(data); cllm_whisper_turbo_model_close(&engine.model); return 1; }
        if (!trial) { expected = body; expected_length = length; }
        else free(body);
    }
    int failed = fwrite(expected, 1, expected_length, stdout) != expected_length || fflush(stdout);
    free(expected); free(data); cllm_whisper_turbo_model_close(&engine.model);
    return failed ? 1 : 0;
#else
    wt_result result = {0}; wt_error error = {0};
    int rc = wt_transcribe(&engine, &req, &result, &error, NULL, NULL);
    char *body = NULL; size_t length = 0; const char *type = NULL;
    if (!rc) rc = wt_render(&req, &result, &body, &length, &type);
    if (!rc && fwrite(body, 1, length, stdout) != length) rc = -1;
    if (!rc && fflush(stdout)) rc = -1;
    clock_gettime(CLOCK_MONOTONIC, &end);
    fprintf(stderr, "{\"asr_windows\":%zu,\"diarization_passes\":%zu,\"seconds\":%.6f,\"success\":%s}\n",
            result.asr_windows, result.diarization_passes,
            end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9, rc ? "false" : "true");
    if (rc && error.message) fprintf(stderr, "%s\n", error.message);
    free(body); wt_result_free(&result); free(data);
    cllm_whisper_turbo_model_close(&engine.model);
    return rc ? 1 : 0;
#endif
}
