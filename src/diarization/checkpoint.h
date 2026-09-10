#ifndef DIAR_CHECKPOINT_H
#define DIAR_CHECKPOINT_H
#include <stddef.h>
#include <stdint.h>
/* Read-only, non-executing reader for the pinned PyTorch ZIP checkpoints.
 * Pickle globals are symbolic strings, never imported or executed. */
typedef struct {
    char name[192];
    int rank, dtype; /* dtype: 4 = float32, 8 = int64 */
    size_t shape[4], count;
    const void *data;
} diar_tensor;
typedef struct {
    void *mapping;
    size_t bytes, count;
    diar_tensor tensors[512];
} diar_checkpoint;
int diar_checkpoint_open(diar_checkpoint *c, const char *path);
void diar_checkpoint_close(diar_checkpoint *c);
const diar_tensor *diar_tensor_find(const diar_checkpoint *c, const char *name);
/* Returns NULL unless dtype, rank and every dimension match. */
const float *diar_weights(const diar_checkpoint *c, const char *name, int rank,
                          const size_t *shape);
int diar_npz_read(const char *path, const char *name, int rank, const size_t *shape, double *out);
#endif
