#ifndef DIAR_PIPELINE_H
#define DIAR_PIPELINE_H
#include <stddef.h>
typedef int (*diar_cancel)(void *);
typedef struct {
    double start, end;
    int speaker;
} diar_interval;
typedef struct {
    diar_interval *segments, *exclusive;
    size_t count, exclusive_count, chunks, training_embeddings;
    int speakers, *chunk_counts;
    float centroids[32 * 256];
} diar_result;
int diar_run(const char *, const float *, size_t, int, diar_result *, diar_cancel, void *);
void diar_result_free(diar_result *);
#endif
