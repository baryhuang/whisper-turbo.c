/* SPDX-License-Identifier: Apache-2.0
 * Native C implementation of the pinned pyannote PyanNet and WeSpeaker
 * inference graphs. See docs/diarization.md and THIRD_PARTY_DIARIZATION.md. */
#include "network.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
__attribute__((target("avx2,fma"))) static float dot_avx(const float *a, const float *b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    const size_t limit = n & ~(size_t)7;
    for (size_t i = 0; i < limit; i += 8)
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    float v[8];
    _mm256_storeu_ps(v, sum);
    float r = 0;
    for (int j = 0; j < 8; ++j)
        r += v[j];
    for (size_t j = 0; j < n - limit; ++j)
        r += a[limit + j] * b[limit + j];
    return r;
}
__attribute__((target("avx512f,fma"))) static float dot_avx512(const float *a, const float *b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    const size_t limit = n & ~(size_t)15;
    for (size_t i = 0; i < limit; i += 16)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), sum);
    float r = _mm512_reduce_add_ps(sum);
    for (size_t j = 0; j < n - limit; ++j) r += a[limit+j]*b[limit+j];
    return r;
}
#endif
float diar_dot(const float *a, const float *b, size_t n) {
#if defined(__aarch64__)
    float32x4_t s = vdupq_n_f32(0);
    const size_t limit = n & ~(size_t)3;
    for (size_t i = 0; i < limit; i += 4)
        s = vfmaq_f32(s, vld1q_f32(a + i), vld1q_f32(b + i));
    float r = vaddvq_f32(s);
    for (size_t j = 0; j < n - limit; ++j)
        r += a[limit + j] * b[limit + j];
    return r;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    /* Opt-in until full diarization parity is established on the target CPU. */
    /* Each inference thread resolves startup configuration once. getenv in
       every short dot product otherwise erases much of the SIMD benefit. */
    static _Thread_local int selected = -1;
    if (selected < 0) {
        const char *simd = getenv("WHISPER_DIAR_SIMD");
        selected = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") ? 1 : 0;
        if (simd && !strcmp(simd, "avx512") && __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("fma")) selected = 2;
    }
    if (selected == 2)
        return dot_avx512(a, b, n);
    if (selected == 1)
        return dot_avx(a, b, n);
#endif
    float sum = 0;
    for (size_t k = 0; k < n; ++k)
        sum += a[k] * b[k];
    return sum;
}
static const float *w(const diar_checkpoint *m, const char *name, int rank, size_t a, size_t b,
                      size_t c, size_t d) {
    size_t s[] = {a, b, c, d};
    const float *p = diar_weights(m, name, rank, s);
    if (!p)
        fprintf(stderr, "missing or incompatible tensor: %s\n", name);
    return p;
}
static float leaky(float x) {
    return x >= 0 ? x : 0.01f * x;
}
static float sigmoid(float x) {
    return x >= 0 ? 1 / (1 + expf(-x)) : expf(x) / (1 + expf(x));
}
static const float *named_weight(const diar_checkpoint *m, const char *base, const char *suffix,
                                 int rank, size_t a, size_t b, size_t c, size_t d) {
    char name[256];
    size_t prefix = strlen(base), tail = strlen(suffix);
    if (prefix >= sizeof(name) || tail >= sizeof(name) - prefix) {
        fprintf(stderr, "tensor name too long\n");
        return NULL;
    }
    memcpy(name, base, prefix);
    memcpy(name + prefix, suffix, tail + 1);
    return w(m, name, rank, a, b, c, d);
}
static int norm(float *x, int t, int c, const diar_checkpoint *m, const char *base) {
    const float *a = named_weight(m, base, ".weight", 1, c, 0, 0, 0);
    const float *b = named_weight(m, base, ".bias", 1, c, 0, 0, 0);
    if (!a || !b)
        return -1;
    for (int j = 0; j < c; ++j) {
        double mean = 0, var = 0;
        for (int i = 0; i < t; ++i)
            mean += x[(size_t)i * c + j];
        mean /= t;
        for (int i = 0; i < t; ++i) {
            double z = x[(size_t)i * c + j] - mean;
            var += z * z;
        }
        float scale = a[j] / sqrtf((float)(var / t) + 1e-5f);
        for (int i = 0; i < t; ++i)
            x[(size_t)i * c + j] = (x[(size_t)i * c + j] - (float)mean) * scale + b[j];
    }
    return 0;
}
static float *conv1_pool(const float *x, int t, int ci, int co, int kernel, int stride,
                         const float *weight, const float *bias, int absolute, int *out_t) {
    int ct = (t - kernel) / stride + 1, nt = ct / 3;
    *out_t = nt;
    if (nt <= 0)
        return NULL;
    float *y = malloc((size_t)nt * co * sizeof(float));
    if (!y)
        return NULL;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nt; ++i) {
        float patch[3][400];
        for (int p = 0; p < 3; ++p)
            for (int c = 0; c < ci; ++c)
                for (int k = 0; k < kernel; ++k)
                    patch[p][c * kernel + k] = x[(size_t)((i * 3 + p) * stride + k) * ci + c];
        for (int o = 0; o < co; ++o) {
            float best = -INFINITY;
            for (int p = 0; p < 3; ++p) {
                float z =
                    diar_dot(patch[p], weight + (size_t)o * ci * kernel, (size_t)ci * kernel) +
                    (bias ? bias[o] : 0);
                if (absolute)
                    z = fabsf(z);
                if (z > best)
                    best = z;
            }
            y[(size_t)i * co + o] = best;
        }
    }
    return y;
}
static int linear(const diar_checkpoint *m, const char *base, const float *x, float *y, int t,
                  int ci, int co, int activate) {
    const float *a = named_weight(m, base, ".weight", 2, co, ci, 0, 0);
    const float *b = named_weight(m, base, ".bias", 1, co, 0, 0, 0);
    if (!a || !b)
        return -1;
    for (int i = 0; i < t; ++i)
        for (int j = 0; j < co; ++j) {
            float z = diar_dot(x + (size_t)i * ci, a + (size_t)j * ci, ci) + b[j];
            y[(size_t)i * co + j] = activate ? leaky(z) : z;
        }
    return 0;
}
int diar_segment(const diar_checkpoint *m, const float *audio, float *prob) {
    float *x = malloc(DIAR_SAMPLES * sizeof(float)), *y = NULL;
    if (!x)
        return -1;
    memcpy(x, audio, DIAR_SAMPLES * sizeof(float));
    if (norm(x, DIAR_SAMPLES, 1, m, "sincnet.wav_norm1d"))
        goto fail;
    const float *low = w(m, "sincnet.conv1d.0.filterbank.low_hz_", 2, 40, 1, 0, 0),
                *band = w(m, "sincnet.conv1d.0.filterbank.band_hz_", 2, 40, 1, 0, 0);
    const float *win = w(m, "sincnet.conv1d.0.filterbank.window_", 1, 125, 0, 0, 0),
                *n = w(m, "sincnet.conv1d.0.filterbank.n_", 2, 1, 125, 0, 0);
    if (!low || !band || !win || !n)
        goto fail;
    float filters[80 * 251];
    for (int c = 0; c < 40; ++c) {
        float lo = 50 + fabsf(low[c]), hi = fminf(8000, lo + 50 + fabsf(band[c])), bw = hi - lo;
        if (bw <= 0)
            goto fail;
        for (int k = 0; k < 125; ++k) {
            float a = (sinf(hi * n[k]) - sinf(lo * n[k])) / n[k] * win[k] / bw;
            float b = (cosf(lo * n[k]) - cosf(hi * n[k])) / n[k] * win[k] / bw;
            filters[c * 251 + k] = filters[c * 251 + 250 - k] = a;
            filters[(c + 40) * 251 + k] = b;
            filters[(c + 40) * 251 + 250 - k] = -b;
        }
        filters[c * 251 + 125] = 1;
        filters[(c + 40) * 251 + 125] = 0;
    }
    int t = DIAR_SAMPLES, ci = 1;
    for (int layer = 0; layer < 3; ++layer) {
        char name[192];
        int co = layer ? 60 : 80, kernel = layer ? 5 : 251, nt;
        const float *weight = filters, *bias = NULL;
        if (layer) {
            snprintf(name, sizeof(name), "sincnet.conv1d.%d.weight", layer);
            weight = w(m, name, 3, co, ci, kernel, 0);
            snprintf(name, sizeof(name), "sincnet.conv1d.%d.bias", layer);
            bias = w(m, name, 1, co, 0, 0, 0);
            if (!weight || !bias)
                goto fail;
        }
        y = conv1_pool(x, t, ci, co, kernel, layer ? 1 : 10, weight, bias, !layer, &nt);
        if (!y)
            goto fail;
        free(x);
        x = y;
        y = NULL;
        t = nt;
        ci = co;
        snprintf(name, sizeof(name), "sincnet.norm1d.%d", layer);
        if (norm(x, t, ci, m, name))
            goto fail;
        for (int i = 0; i < t * ci; ++i)
            x[i] = leaky(x[i]);
    }
    if (t != DIAR_FRAMES)
        goto fail;
    for (int layer = 0; layer < 4; ++layer) {
        y = calloc((size_t)t * 256, sizeof(float));
        if (!y)
            goto fail;
        for (int rev = 0; rev < 2; ++rev) {
            char name[192];
            const char *suffix = rev ? "_reverse" : "";
            snprintf(name, sizeof(name), "lstm.weight_ih_l%d%s", layer, suffix);
            const float *wi = w(m, name, 2, 512, ci, 0, 0);
            snprintf(name, sizeof(name), "lstm.weight_hh_l%d%s", layer, suffix);
            const float *wh = w(m, name, 2, 512, 128, 0, 0);
            snprintf(name, sizeof(name), "lstm.bias_ih_l%d%s", layer, suffix);
            const float *bi = w(m, name, 1, 512, 0, 0, 0);
            snprintf(name, sizeof(name), "lstm.bias_hh_l%d%s", layer, suffix);
            const float *bh = w(m, name, 1, 512, 0, 0, 0);
            if (!wi || !wh || !bi || !bh)
                goto fail;
            /* Input projections have no time dependency. Batch these across
               the window, leaving only recurrent state updates sequential.
               Scratch is bounded to this fixed ten-second segmentation window. */
            float *input_gates = NULL;
            const char *batch = getenv("WHISPER_DIAR_BATCH_INPUT");
            if (batch && !strcmp(batch, "1")) {
                input_gates = malloc((size_t)t * 512 * sizeof(float));
                if (!input_gates) goto fail;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int at = 0; at < t; ++at)
                    for (int j = 0; j < 512; ++j)
                        input_gates[(size_t)at * 512 + j] =
                            diar_dot(x + (size_t)at * ci, wi + (size_t)j * ci, ci);
            }
            float h[128] = {0}, cell[128] = {0}, g[512];
            for (int step = 0; step < t; ++step) {
                int at = rev ? t - 1 - step : step;
                for (int j = 0; j < 512; ++j)
                    g[j] = (input_gates ? input_gates[(size_t)at * 512 + j] :
                            diar_dot(x + (size_t)at * ci, wi + (size_t)j * ci, ci)) +
                           diar_dot(h, wh + (size_t)j * 128, 128) + bi[j] + bh[j];
                for (int j = 0; j < 128; ++j) {
                    cell[j] = sigmoid(g[128 + j]) * cell[j] + sigmoid(g[j]) * tanhf(g[256 + j]);
                    h[j] = sigmoid(g[384 + j]) * tanhf(cell[j]);
                    y[(size_t)at * 256 + rev * 128 + j] = h[j];
                }
            }
            free(input_gates);
        }
        free(x);
        x = y;
        y = NULL;
        ci = 256;
    }
    y = malloc((size_t)t * 128 * sizeof(float));
    if (!y || linear(m, "linear.0", x, y, t, 256, 128, 1))
        goto fail;
    free(x);
    x = y;
    y = NULL;
    y = malloc((size_t)t * 128 * sizeof(float));
    if (!y || linear(m, "linear.1", x, y, t, 128, 128, 1))
        goto fail;
    free(x);
    x = y;
    y = NULL;
    if (linear(m, "classifier", x, prob, t, 128, 7, 0))
        goto fail;
    for (int i = 0; i < t; ++i) {
        float max = prob[i * 7], sum = 0;
        for (int j = 1; j < 7; ++j)
            max = fmaxf(max, prob[i * 7 + j]);
        for (int j = 0; j < 7; ++j) {
            prob[i * 7 + j] = expf(prob[i * 7 + j] - max);
            sum += prob[i * 7 + j];
        }
        if (!isfinite(sum) || sum <= 0)
            goto fail;
        for (int j = 0; j < 7; ++j)
            prob[i * 7 + j] /= sum;
    }
    free(x);
    return 0;
fail:
    free(x);
    free(y);
    return -1;
}
void diar_powerset(const float *p, float *mask, size_t frames) {
    static const unsigned bits[] = {0, 1, 2, 4, 3, 5, 6};
    for (size_t t = 0; t < frames; ++t) {
        int k = 0;
        for (int j = 1; j < 7; ++j)
            if (p[t * 7 + j] > p[t * 7 + k])
                k = j;
        for (int s = 0; s < 3; ++s)
            mask[t * 3 + s] = (float)((bits[k] >> s) & 1);
    }
}
/* Kaldi-compatible fbank: snip_edges, 25 ms Hamming, 10 ms hop,
 * DC removal, pre-emphasis 0.97, power spectrum, 80 mel bins, no dither. */
static void fft(float *re, float *im) {
    for (unsigned i = 1, j = 0; i < 512; ++i) {
        unsigned bit = 256;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }
    for (unsigned len = 2; len <= 512; len *= 2)
        for (unsigned start = 0; start < 512; start += len)
            for (unsigned k = 0; k < len / 2; ++k) {
                float phase = -6.283185307179586f * k / len, c = cosf(phase), s = sinf(phase);
                unsigned a = start + k, b = a + len / 2;
                float vr = re[b] * c - im[b] * s, vi = re[b] * s + im[b] * c;
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] += vr;
                im[a] += vi;
            }
}
int diar_fbank(const float *audio, size_t samples, float *out) {
    if (samples < 400 || samples > DIAR_SAMPLES)
        return -1;
    int frames = (int)((samples - 400) / 160 + 1);
    float mel[80][256], window[400];
    float low = 1127 * logf(1 + 20.f / 700), high = 1127 * logf(1 + 8000.f / 700),
          delta = (high - low) / 81;
    for (int m = 0; m < 80; ++m)
        for (int k = 0; k < 256; ++k) {
            float f = 1127 * logf(1 + (k * 16000.f / 512) / 700);
            mel[m][k] =
                fmaxf(0, fminf((f - low - m * delta) / delta, (low + (m + 2) * delta - f) / delta));
        }
    for (int j = 0; j < 400; ++j)
        window[j] = 0.54f - 0.46f * cosf(6.283185307179586f * j / 399);
    for (int t = 0; t < frames; ++t) {
        float re[512] = {0}, im[512] = {0}, raw[400];
        double mean = 0;
        for (int j = 0; j < 400; ++j) {
            raw[j] = audio[t * 160 + j] * 32768;
            mean += raw[j];
        }
        mean /= 400;
        for (int j = 0; j < 400; ++j)
            raw[j] -= (float)mean;
        for (int j = 0; j < 400; ++j)
            re[j] = (raw[j] - 0.97f * raw[j ? j - 1 : 0]) * window[j];
        fft(re, im);
        float power[256];
        for (int k = 0; k < 256; ++k)
            power[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < 80; ++m)
            out[(size_t)t * 80 + m] = logf(fmaxf(FLT_EPSILON, diar_dot(mel[m], power, 256)));
    }
    for (int m = 0; m < 80; ++m) {
        double mean = 0;
        for (int t = 0; t < frames; ++t)
            mean += out[(size_t)t * 80 + m];
        mean /= frames;
        for (int t = 0; t < frames; ++t)
            out[(size_t)t * 80 + m] -= (float)mean;
    }
    return frames;
}
/* Layout [frequency][time][channel]. Patch tiling bounds scratch irrespective
 * of recording length; only one ten-second window is processed at a time. */
static float *conv2(const diar_checkpoint *m, const char *base, const char *bn, const float *x,
                    int h, int t, int ci, int co, int kernel, int stride, int relu, int *oh,
                    int *ot) {
    const float *weight = named_weight(m, base, ".weight", 4, co, ci, kernel, kernel);
    const float *gamma = named_weight(m, bn, ".weight", 1, co, 0, 0, 0);
    const float *beta = named_weight(m, bn, ".bias", 1, co, 0, 0, 0);
    const float *mean = named_weight(m, bn, ".running_mean", 1, co, 0, 0, 0);
    const float *var = named_weight(m, bn, ".running_var", 1, co, 0, 0, 0);
    if (!weight || !gamma || !beta || !mean || !var)
        return NULL;
    *oh = (h - 1) / stride + 1;
    *ot = (t - 1) / stride + 1;
    int rows = *oh * *ot, k = ci * kernel * kernel, pad = kernel / 2;
    float *y = malloc((size_t)rows * co * sizeof(float));
    if (!y)
        return NULL;
    float scale[256], offset[256];
    for (int c = 0; c < co; ++c) {
        scale[c] = gamma[c] / sqrtf(var[c] + 1e-5f);
        offset[c] = beta[c] - mean[c] * scale[c];
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < rows; ++row) {
        float patch[2304];
        int f = row / (*ot), time = row % (*ot), at = 0;
        for (int c = 0; c < ci; ++c)
            for (int a = 0; a < kernel; ++a)
                for (int b = 0; b < kernel; ++b) {
                    int fh = f * stride + a - pad, tt = time * stride + b - pad;
                    patch[at++] = (fh >= 0 && fh < h && tt >= 0 && tt < t)
                                      ? x[((size_t)fh * t + tt) * ci + c]
                                      : 0;
                }
        for (int c = 0; c < co; ++c) {
            float z = diar_dot(patch, weight + (size_t)c * k, k) * scale[c] + offset[c];
            y[(size_t)row * co + c] = relu ? fmaxf(z, 0) : z;
        }
    }
    return y;
}
int diar_embed(const diar_checkpoint *m, const float *audio, const float *masks, float *emb) {
    float *fbank = malloc(998 * 80 * sizeof(float)), *x = NULL, *y = NULL, *z = NULL,
          *shortcut = NULL;
    if (!fbank)
        return -1;
    int t = diar_fbank(audio, DIAR_SAMPLES, fbank), h = 80, ci = 1, nh, nt;
    if (t != 998)
        goto fail;
    x = malloc((size_t)t * 80 * sizeof(float));
    if (!x)
        goto fail;
    for (int f = 0; f < 80; ++f)
        for (int i = 0; i < t; ++i)
            x[(size_t)f * t + i] = fbank[(size_t)i * 80 + f];
    free(fbank);
    fbank = NULL;
    y = conv2(m, "resnet.conv1", "resnet.bn1", x, h, t, 1, 32, 3, 1, 1, &nh, &nt);
    if (!y)
        goto fail;
    free(x);
    x = y;
    y = NULL;
    ci = 32;
    const int blocks[] = {3, 4, 6, 3};
    for (int stage = 0; stage < 4; ++stage)
        for (int b = 0; b < blocks[stage]; ++b) {
            int co = 32 << stage, stride = stage && b == 0 ? 2 : 1;
            char base[192], bn[192];
            snprintf(base, sizeof(base), "resnet.layer%d.%d.conv1", stage + 1, b);
            snprintf(bn, sizeof(bn), "resnet.layer%d.%d.bn1", stage + 1, b);
            y = conv2(m, base, bn, x, h, t, ci, co, 3, stride, 1, &nh, &nt);
            if (!y)
                goto fail;
            snprintf(base, sizeof(base), "resnet.layer%d.%d.conv2", stage + 1, b);
            snprintf(bn, sizeof(bn), "resnet.layer%d.%d.bn2", stage + 1, b);
            int zh, zt;
            z = conv2(m, base, bn, y, nh, nt, co, co, 3, 1, 0, &zh, &zt);
            if (!z)
                goto fail;
            free(y);
            y = NULL;
            if (stride != 1 || ci != co) {
                snprintf(base, sizeof(base), "resnet.layer%d.%d.shortcut.0", stage + 1, b);
                snprintf(bn, sizeof(bn), "resnet.layer%d.%d.shortcut.1", stage + 1, b);
                shortcut = conv2(m, base, bn, x, h, t, ci, co, 1, stride, 0, &zh, &zt);
                if (!shortcut)
                    goto fail;
            }
            const float *skip = shortcut ? shortcut : x;
            for (size_t j = 0; j < (size_t)nh * nt * co; ++j)
                z[j] = fmaxf(0, z[j] + skip[j]);
            free(shortcut);
            shortcut = NULL;
            free(x);
            x = z;
            z = NULL;
            h = nh;
            t = nt;
            ci = co;
        }
    if (h != 10 || t != 125 || ci != 256)
        goto fail;
    for (int s = 0; s < 3; ++s) {
        float weights[125], stats[5120];
        double v1 = 1e-8, v2 = 0;
        for (int i = 0; i < t; ++i) {
            weights[i] = masks[(size_t)(i * DIAR_FRAMES / t) * 3 + s];
            v1 += weights[i];
            v2 += weights[i] * weights[i];
        }
        if (v1 < 0.5) {
            for (int j = 0; j < 256; ++j)
                emb[s * 256 + j] = NAN;
            continue;
        }
        for (int c = 0; c < 256; ++c)
            for (int f = 0; f < 10; ++f) {
                int d = c * 10 + f;
                double mean = 0, var = 0;
                for (int i = 0; i < t; ++i)
                    mean += x[((size_t)f * t + i) * 256 + c] * weights[i];
                mean /= v1;
                for (int i = 0; i < t; ++i) {
                    double a = x[((size_t)f * t + i) * 256 + c] - mean;
                    var += a * a * weights[i];
                }
                stats[d] = (float)mean;
                stats[2560 + d] = (float)sqrt(var / (v1 - v2 / v1 + 1e-8));
            }
        if (linear(m, "resnet.seg_1", stats, emb + s * 256, 1, 5120, 256, 0))
            goto fail;
        for (int j = 0; j < 256; ++j)
            if (!isfinite(emb[s * 256 + j]))
                goto fail;
    }
    free(x);
    return 0;
fail:
    free(fbank);
    free(x);
    free(y);
    free(z);
    free(shortcut);
    return -1;
}
