#ifndef WT_API_H
#define WT_API_H
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#define WT_UPLOAD_LIMIT 25000000U
#define WT_TEXT_LIMIT 262144U
#define WT_AUDIO_LIMIT (16000U * 120U)
typedef struct {
    int status;
    const char *message, *param, *code;
} wt_error;
typedef struct {
    const unsigned char *file;
    size_t file_size;
    char language[4]; /* Empty means model-based language detection. */
    int plain_text;
} wt_request;
typedef struct {
    unsigned char *text;
    size_t length;
} wt_result;
typedef int (*wt_cancel)(void *);
typedef int (*wt_backend)(void *, const wt_request *, wt_result *, wt_error *, wt_cancel, void *);
typedef struct {
    const char *host, *api_key;
    unsigned port, timeout_seconds, upload_seconds;
} wt_server_options;

int wt_fail(wt_error *e, int status, const char *message, const char *param, const char *code);
int wt_boundary(const char *content_type, char out[71]);
int wt_multipart(const unsigned char *body, size_t length, const char *boundary,
                 wt_request *request, wt_error *error);
int wt_wav(const unsigned char *data, size_t length, const unsigned char **pcm, size_t *samples,
           wt_error *error);
/* Allocated UTF-8 JSON string, including surrounding quotes. */
char *wt_json_string(const unsigned char *data, size_t length);
int wt_serve(const wt_server_options *, wt_backend, void *, atomic_int *stop);
#endif
