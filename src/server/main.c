#define _POSIX_C_SOURCE 200809L
#include "inference.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "signal cancellation requires lock-free atomic int");
static atomic_int stopping;
static void on_signal(int sig) {
    (void)sig;
    stopping = 1;
}
static unsigned number(const char *value, unsigned maximum) {
    if (!value || !*value)
        return 0;
    unsigned n = 0;
    for (; *value; ++value) {
        if (*value < '0' || *value > '9' || n > maximum / 10)
            return 0;
        n = n * 10 + (unsigned)(*value - '0');
        if (n > maximum)
            return 0;
    }
    return n;
}
int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "usage: %s MODEL.whtrbo [PORT [BIND_HOST]]\n"
                "Environment: WHISPER_API_KEY, OMP_NUM_THREADS (1-8), WHISPER_REQUEST_TIMEOUT "
                "(1-3600 seconds), WHISPER_DIARIZATION_MODELS (checkpoint directory).\n",
                argv[0]);
        return 2;
    }
    wt_server_options options = {.host = argc == 4 ? argv[3] : "127.0.0.1",
                                 .port = argc >= 3 ? number(argv[2], 65535) : 8080,
                                 .api_key = getenv("WHISPER_API_KEY"),
                                 .timeout_seconds =
                                     getenv("WHISPER_REQUEST_TIMEOUT")
                                         ? number(getenv("WHISPER_REQUEST_TIMEOUT"), 3600)
                                         : 300,
                                 .upload_seconds = 30};
    wt_engine engine = {.threads =
                            getenv("OMP_NUM_THREADS") ? number(getenv("OMP_NUM_THREADS"), 8) : 8};
    engine.model.image.fd = -1;
    engine.diarization_directory = getenv("WHISPER_DIARIZATION_MODELS");
    if (!options.port || !options.timeout_seconds || !engine.threads ||
        (options.api_key && strlen(options.api_key) > 1024)) {
        fprintf(stderr, "Invalid port, thread count, timeout, or API key length.\n");
        return 2;
    }
    if (strcmp(options.host, "127.0.0.1") && (!options.api_key || !*options.api_key)) {
        fprintf(stderr, "WHISPER_API_KEY is required for non-loopback binding. Use a TLS reverse "
                        "proxy for remote access.\n");
        return 2;
    }
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    action.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &action, NULL);
    if (cllm_whisper_turbo_model_open(argv[1], &engine.model)) {
        fprintf(stderr, "Cannot load Whisper Turbo model.\n");
        return 2;
    }
#ifndef _OPENMP
    fprintf(stderr, "Built without OpenMP; inference uses one thread.\n");
#endif
    int rc = wt_serve(&options, wt_transcribe, &engine, &stopping);
    cllm_whisper_turbo_model_close(&engine.model);
    return rc ? 1 : 0;
}
