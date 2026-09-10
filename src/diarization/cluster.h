#ifndef DIAR_CLUSTER_H
#define DIAR_CLUSTER_H
#include <stddef.h>
#define DIAR_MAX_EMBEDDINGS 336
typedef struct {
    double mean1[256], mean2[128], lda[256 * 128], mu[128], tr[128 * 128], phi[128];
} diar_plda;
int diar_plda_open(diar_plda *p, const char *transform, const char *plda);
void diar_plda_apply(const diar_plda *p, const float *embedding, double *out);
/* Returns the number of centroids, or -1. N bounded to protect AHC workspace.
 * Output centroids are in the original 256-D embedding space. */
int diar_cluster(const diar_plda *p, const float *embeddings, size_t n, float *centroids);
int diar_vbx(const double *x, const double *phi, size_t n, size_t speakers, const int *labels,
             double *q, double *prior);
#endif
