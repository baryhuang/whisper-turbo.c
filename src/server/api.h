#ifndef WT_API_H
#define WT_API_H
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include "../audio_limits.h"

#define WT_UPLOAD_LIMIT 25000000U
#define WT_TEXT_LIMIT 262144U
#define WT_AUDIO_LIMIT WT_MAX_AUDIO_SAMPLES
#define WT_SEGMENT_LIMIT 512U
typedef struct {
    int status;
    const char *message, *param, *code;
} wt_error;
typedef struct {
    const unsigned char *file;
    size_t file_size;
    char language[4]; /* Empty means model-based language detection. */
    int plain_text;
    int diarize, diarized_json, stream, chunk_auto;
    unsigned name_count, reference_count;
    char names[4][64];
    const unsigned char *references[4];
    size_t reference_lengths[4];
} wt_request;
typedef struct {
    double start, end;
    char speaker[64];
    unsigned char *text;
    size_t length;
} wt_segment;
typedef struct {
    size_t offset, length;
    double start, end;
} wt_word;
typedef struct {
    unsigned char *text;
    size_t length;
    double duration;
    wt_segment *segments;
    size_t segment_count;
    /* Internal alignment metadata; not serialized as an API field. */
    wt_word *words;
    size_t word_count, asr_windows, diarization_passes;
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
void wt_result_free(wt_result *);
/* Bounded complete response; SSE events are buffered until inference completes.
 */
int wt_render(const wt_request *, const wt_result *, char **, size_t *, const char **);
int wt_reference_wav(const unsigned char *, size_t, unsigned char **, size_t *, wt_error *);
int wt_serve(const wt_server_options *, wt_backend, void *, atomic_int *stop);
#endif
