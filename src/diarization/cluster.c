/* SPDX-License-Identifier: Apache-2.0
 * Centroid-linkage initialization and PLDA/VBx with Community-1 parameters.
 * Algorithm attribution: pyannote.audio and BUTSpeechFIT/VBx (Apache-2.0).
 * No stochastic initialization or external numerical runtime is used. */
#include "cluster.h"
#include "checkpoint.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
int diar_plda_open(diar_plda *p, const char *transform, const char *plda) {
    if (diar_npz_read(transform, "mean1.npy", 1, (size_t[]){256}, p->mean1) ||
        diar_npz_read(transform, "mean2.npy", 1, (size_t[]){128}, p->mean2) ||
        diar_npz_read(transform, "lda.npy", 2, (size_t[]){256, 128}, p->lda) ||
        diar_npz_read(plda, "mu.npy", 1, (size_t[]){128}, p->mu) ||
        diar_npz_read(plda, "tr.npy", 2, (size_t[]){128, 128}, p->tr) ||
        diar_npz_read(plda, "psi.npy", 1, (size_t[]){128}, p->phi))
        return -1;
    /* Original PLDA transform already diagonalizes between/within covariance:
     * tr W tr^T = I and tr B tr^T = diag(psi). Sorting those rows is equivalent
     * to re-solving eigh(B,W), up to eigenvector signs (VBx is sign-invariant). */
    for (int i = 0; i < 128; ++i) {
        int best = i;
        for (int j = i + 1; j < 128; ++j)
            if (p->phi[j] > p->phi[best])
                best = j;
        double a = p->phi[i];
        p->phi[i] = p->phi[best];
        p->phi[best] = a;
        for (int d = 0; d < 128; ++d) {
            a = p->tr[i * 128 + d];
            p->tr[i * 128 + d] = p->tr[best * 128 + d];
            p->tr[best * 128 + d] = a;
        }
        if (!isfinite(p->phi[i]) || p->phi[i] <= 0)
            return -1;
    }
    return 0;
}
void diar_plda_apply(const diar_plda *p, const float *e, double *out) {
    double a[256], b[128], norm = 0;
    for (int d = 0; d < 256; ++d) {
        a[d] = e[d] - p->mean1[d];
        norm += a[d] * a[d];
    }
    norm = sqrt(norm);
    for (int d = 0; d < 256; ++d)
        a[d] *= 16 / fmax(norm, 1e-30);
    norm = 0;
    for (int j = 0; j < 128; ++j) {
        double s = 0;
        for (int d = 0; d < 256; ++d)
            s += a[d] * p->lda[d * 128 + j];
        b[j] = s - p->mean2[j];
        norm += b[j] * b[j];
    }
    norm = sqrt(norm);
    for (int j = 0; j < 128; ++j)
        b[j] = sqrt(128) * b[j] / fmax(norm, 1e-30) - p->mu[j];
    for (int j = 0; j < 128; ++j) {
        out[j] = 0;
        for (int d = 0; d < 128; ++d)
            out[j] += b[d] * p->tr[j * 128 + d];
    }
}
int diar_vbx(const double *x, const double *phi, size_t n, size_t k, const int *labels, double *q,
             double *prior) {
    if (!n || n > DIAR_MAX_EMBEDDINGS || !k || k > n)
        return -1;
    double *alpha = calloc(k * 128, sizeof(double)), *inv = calloc(k * 128, sizeof(double)),
           *rho = malloc(n * 128 * sizeof(double)), *g = calloc(n, sizeof(double));
    if (!alpha || !inv || !rho || !g) {
        free(alpha);
        free(inv);
        free(rho);
        free(g);
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        if (labels[i] < 0 || (size_t)labels[i] >= k)
            goto fail;
        for (int d = 0; d < 128; ++d) {
            if (!isfinite(phi[d]) || phi[d] <= 0 || !isfinite(x[i * 128 + d]))
                goto fail;
            rho[i * 128 + d] = x[i * 128 + d] * sqrt(phi[d]);
            g[i] -= 0.5 * x[i * 128 + d] * x[i * 128 + d];
        }
        g[i] -= 64 * log(6.283185307179586);
        for (size_t j = 0; j < k; ++j)
            q[i * k + j] = (j == (size_t)labels[i] ? exp(7) : 1) / (exp(7) + k - 1);
    }
    for (size_t j = 0; j < k; ++j)
        prior[j] = 1.0 / k;
    double previous = -INFINITY;
    for (int iter = 0; iter < 20; ++iter) {
        for (size_t j = 0; j < k; ++j) {
            double sum = 0;
            for (size_t i = 0; i < n; ++i)
                sum += q[i * k + j];
            for (int d = 0; d < 128; ++d) {
                double s = 0;
                for (size_t i = 0; i < n; ++i)
                    s += q[i * k + j] * rho[i * 128 + d];
                inv[j * 128 + d] = 1 / (1 + 0.07 / 0.8 * sum * phi[d]);
                alpha[j * 128 + d] = 0.07 / 0.8 * inv[j * 128 + d] * s;
            }
        }
        double obj = 0;
        for (size_t i = 0; i < n; ++i) {
            double max = -INFINITY, sum = 0;
            for (size_t j = 0; j < k; ++j) {
                double z = g[i];
                for (int d = 0; d < 128; ++d) {
                    double a = alpha[j * 128 + d];
                    z += rho[i * 128 + d] * a - 0.5 * (inv[j * 128 + d] + a * a) * phi[d];
                }
                q[i * k + j] = 0.07 * z + log(prior[j] + 1e-8);
                max = fmax(max, q[i * k + j]);
            }
            for (size_t j = 0; j < k; ++j)
                sum += exp(q[i * k + j] - max);
            double logsum = max + log(sum);
            obj += logsum;
            for (size_t j = 0; j < k; ++j)
                q[i * k + j] = exp(q[i * k + j] - logsum);
        }
        for (size_t j = 0; j < k; ++j) {
            prior[j] = 0;
            for (size_t i = 0; i < n; ++i)
                prior[j] += q[i * k + j] / n;
        }
        for (size_t j = 0; j < k * 128; ++j)
            obj += 0.4 * (log(inv[j]) - inv[j] - alpha[j] * alpha[j] + 1);
        if (iter && obj - previous < 1e-4)
            break;
        previous = obj;
    }
    free(alpha);
    free(inv);
    free(rho);
    free(g);
    return 0;
fail:
    free(alpha);
    free(inv);
    free(rho);
    free(g);
    return -1;
}
static void label_tree(int node, int label, int leaves, const int *left, const int *right,
                       int *labels) {
    if (node < leaves) {
        labels[node] = label;
        return;
    }
    label_tree(left[node], label, leaves, left, right, labels);
    label_tree(right[node], label, leaves, left, right, labels);
}
static void cut_tree(int node, int leaves, const int *left, const int *right, const double *height,
                     int *labels, int *next) {
    if (node < leaves || height[node] <= 0.6) {
        label_tree(node, (*next)++, leaves, left, right, labels);
        return;
    }
    cut_tree(left[node], leaves, left, right, height, labels, next);
    cut_tree(right[node], leaves, left, right, height, labels, next);
}
int diar_cluster(const diar_plda *p, const float *e, size_t n, float *centroids) {
    if (!n || n > DIAR_MAX_EMBEDDINGS)
        return -1;
    if (n == 1) {
        memcpy(centroids, e, 256 * sizeof(float));
        return 1;
    }
    double *cent = calloc(2 * n * 256, sizeof(double)), *dist = calloc(4 * n * n, sizeof(double)),
           *height = calloc(2 * n, sizeof(double));
    int *left = calloc(2 * n, sizeof(int)), *right = calloc(2 * n, sizeof(int)),
        *size = calloc(2 * n, sizeof(int)), *active = calloc(2 * n, sizeof(int)),
        *labels = calloc(n, sizeof(int));
    double *x = malloc(n * 128 * sizeof(double)), *q = NULL, *prior = NULL;
    int result = -1;
    if (!cent || !dist || !height || !left || !right || !size || !active || !labels || !x)
        goto done;
    for (size_t i = 0; i < n; ++i) {
        double norm = 0;
        for (int d = 0; d < 256; ++d) {
            if (!isfinite(e[i * 256 + d]))
                goto done;
            norm += e[i * 256 + d] * e[i * 256 + d];
        }
        if (norm < 1e-30)
            goto done;
        norm = sqrt(norm);
        for (int d = 0; d < 256; ++d)
            cent[i * 256 + d] = e[i * 256 + d] / norm;
        active[i] = size[i] = 1;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < i; ++j) {
            double s = 0;
            for (int d = 0; d < 256; ++d) {
                double a = cent[i * 256 + d] - cent[j * 256 + d];
                s += a * a;
            }
            dist[i * 2 * n + j] = dist[j * 2 * n + i] = s;
        }
    for (size_t node = n; node < 2 * n - 1; ++node) {
        double best = INFINITY;
        int a = -1, b = -1;
        for (size_t i = 0; i < node; ++i)
            if (active[i])
                for (size_t j = 0; j < i; ++j)
                    if (active[j] && dist[i * 2 * n + j] < best) {
                        best = dist[i * 2 * n + j];
                        a = (int)i;
                        b = (int)j;
                    }
        if (a < 0 || b < 0)
            goto done;
        left[node] = a;
        right[node] = b;
        size[node] = size[a] + size[b];
        height[node] = fmax(sqrt(fmax(0, best)), fmax(height[a], height[b]));
        active[a] = active[b] = 0;
        active[node] = 1;
        for (int d = 0; d < 256; ++d)
            cent[node * 256 + d] =
                (size[a] * cent[a * 256 + d] + size[b] * cent[b * 256 + d]) / size[node];
        for (size_t i = 0; i < node; ++i)
            if (active[i]) {
                double s = 0;
                for (int d = 0; d < 256; ++d) {
                    double z = cent[node * 256 + d] - cent[i * 256 + d];
                    s += z * z;
                }
                dist[node * 2 * n + i] = dist[i * 2 * n + node] = s;
            }
    }
    int k = 0;
    cut_tree((int)(2 * n - 2), (int)n, left, right, height, labels, &k);
    q = malloc(n * (size_t)k * sizeof(double));
    prior = malloc((size_t)k * sizeof(double));
    if (!q || !prior)
        goto done;
    for (size_t i = 0; i < n; ++i)
        diar_plda_apply(p, e + i * 256, x + i * 128);
    if (diar_vbx(x, p->phi, n, (size_t)k, labels, q, prior))
        goto done;
    result = 0;
    for (int j = 0; j < k; ++j)
        if (prior[j] > 1e-7) {
            double sum = 0;
            for (size_t i = 0; i < n; ++i)
                sum += q[i * k + j];
            for (int d = 0; d < 256; ++d) {
                double a = 0;
                for (size_t i = 0; i < n; ++i)
                    a += q[i * k + j] * e[i * 256 + d];
                centroids[result * 256 + d] = (float)(a / sum);
            }
            ++result;
        }
done:
    free(cent);
    free(dist);
    free(height);
    free(left);
    free(right);
    free(size);
    free(active);
    free(labels);
    free(x);
    free(q);
    free(prior);
    return result;
}
