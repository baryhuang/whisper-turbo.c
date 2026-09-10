#define _POSIX_C_SOURCE 200809L
#include "audio.h"
#include "cluster.h"
#include "network.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                                   \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
static int kernels(void) {
    float a[257], b[257];
    double expected = 0;
    for (int i = 0; i < 257; ++i) {
        a[i] = sinf(i * 0.37f);
        b[i] = cosf(i * 0.23f);
        expected += (double)a[i] * b[i];
    }
    REQUIRE(fabs(diar_dot(a, b, 257) - expected) < 2e-5);
    /* Exercise vector-width boundaries and every short scalar tail. */
    expected = 0;
    REQUIRE(diar_dot(a, b, 0) == 0);
    for (size_t n = 1; n <= 257; ++n) {
        expected += (double)a[n - 1] * b[n - 1];
        REQUIRE(fabs(diar_dot(a, b, n) - expected) < 2e-5);
    }
    float p[49] = {0}, mask[21];
    for (int i = 0; i < 7; ++i)
        p[i * 7 + i] = 1;
    diar_powerset(p, mask, 7);
    unsigned bits[] = {0, 1, 2, 4, 3, 5, 6};
    for (int i = 0; i < 7; ++i)
        for (int s = 0; s < 3; ++s)
            REQUIRE(mask[i * 3 + s] == ((bits[i] >> s) & 1));
    float *audio = calloc(16000, sizeof(float)), *features = malloc(98 * 80 * sizeof(float));
    REQUIRE(audio && features);
    REQUIRE(diar_fbank(audio, 16000, features) == 98);
    for (int i = 0; i < 98 * 80; ++i)
        REQUIRE(fabsf(features[i]) < 1e-6f);
    for (int i = 0; i < 16000; ++i)
        audio[i] = 0.3f * sinf(i * 6.283185307179586f * 440 / 16000);
    REQUIRE(diar_fbank(audio, 16000, features) == 98);
    for (int m = 0; m < 80; ++m) {
        double sum = 0;
        for (int t = 0; t < 98; ++t) {
            REQUIRE(isfinite(features[t * 80 + m]));
            sum += features[t * 80 + m];
        }
        REQUIRE(fabs(sum / 98) < 2e-6);
    }
    REQUIRE(diar_fbank(audio, 399, features) == -1);
    free(audio);
    free(features);
    return 0;
}
static int clustering(void) {
    double x[6 * 128] = {0}, phi[128], q[12], prior[2], q2[12], prior2[2];
    int labels[] = {0, 0, 0, 1, 1, 1};
    for (int d = 0; d < 128; ++d)
        phi[d] = 1;
    for (int i = 0; i < 6; ++i)
        x[i * 128] = (i < 3 ? 50 : -50);
    REQUIRE(!diar_vbx(x, phi, 6, 2, labels, q, prior));
    for (int i = 0; i < 6; ++i) {
        REQUIRE(fabs(q[i * 2] + q[i * 2 + 1] - 1) < 1e-12);
        REQUIRE(q[i * 2 + labels[i]] > 0.99);
    }
    for (int i = 0; i < 6; ++i)
        x[i * 128] = -x[i * 128];
    REQUIRE(!diar_vbx(x, phi, 6, 2, labels, q2, prior2));
    for (int i = 0; i < 12; ++i)
        REQUIRE(fabs(q[i] - q2[i]) < 1e-12);
    REQUIRE(fabs(prior[0] + prior[1] - 1) < 1e-12);
    REQUIRE(diar_vbx(x, phi, 0, 2, labels, q, prior) == -1);
    return 0;
}
static int invalid_archive(void) {
    char path[] = "/tmp/diar-invalid-XXXXXX";
    int fd = mkstemp(path);
    REQUIRE(fd >= 0);
    unsigned char bytes[128] = {0};
    memcpy(bytes, "PK\003\004", 4);
    REQUIRE(write(fd, bytes, sizeof(bytes)) == (ssize_t)sizeof(bytes));
    close(fd);
    diar_checkpoint c;
    int status = diar_checkpoint_open(&c, path);
    unlink(path);
    REQUIRE(status == -1 && c.mapping == NULL);
    return 0;
}
static int checkpoints(const char *dir) {
    char path[4096], other[4096];
    diar_checkpoint c;
    snprintf(path, sizeof(path), "%s/segmentation-pytorch_model.bin", dir);
    REQUIRE(!diar_checkpoint_open(&c, path));
    const diar_tensor *t = diar_tensor_find(&c, "lstm.weight_ih_l0");
    REQUIRE(t && t->shape[0] == 512 && t->shape[1] == 60);
    const diar_tensor *u = diar_tensor_find(&c, "lstm.weight_ih_l1");
    REQUIRE(u && u->data != t->data);
    REQUIRE(!diar_weights(&c, "lstm.weight_ih_l0", 2, (size_t[]){1, 1}));
    /* A payload mutation must fail ZIP CRC validation, not enter inference. */
    unsigned char *copy = malloc(c.bytes);
    REQUIRE(copy);
    memcpy(copy, c.mapping, c.bytes);
    size_t offset = (const unsigned char *)t->data - (const unsigned char *)c.mapping;
    copy[offset] ^= 1;
    char corrupt[] = "/tmp/diar-corrupt-XXXXXX";
    int fd = mkstemp(corrupt);
    REQUIRE(fd >= 0);
    FILE *f = fdopen(fd, "wb");
    REQUIRE(f);
    REQUIRE(fwrite(copy, 1, c.bytes, f) == c.bytes);
    REQUIRE(!fclose(f));
    free(copy);
    diar_checkpoint rejected;
    int status = diar_checkpoint_open(&rejected, corrupt);
    unlink(corrupt);
    REQUIRE(status == -1 && rejected.mapping == NULL);
    float *audio = calloc(DIAR_SAMPLES, sizeof(float)),
          *p = malloc(DIAR_FRAMES * 7 * sizeof(float));
    REQUIRE(audio && p);
    REQUIRE(!diar_segment(&c, audio, p));
    for (int f = 0; f < DIAR_FRAMES; ++f) {
        double sum = 0;
        for (int k = 0; k < 7; ++k) {
            REQUIRE(isfinite(p[f * 7 + k]) && p[f * 7 + k] >= 0 && p[f * 7 + k] <= 1);
            sum += p[f * 7 + k];
        }
        REQUIRE(fabs(sum - 1) < 1e-6);
    }
    float silence[DIAR_FRAMES * 3];
    diar_powerset(p, silence, DIAR_FRAMES);
    for (int i = 0; i < DIAR_FRAMES * 3; ++i)
        REQUIRE(silence[i] == 0);
    free(audio);
    free(p);
    diar_checkpoint_close(&c);
    snprintf(path, sizeof(path), "%s/embedding-pytorch_model.bin", dir);
    REQUIRE(!diar_checkpoint_open(&c, path));
    t = diar_tensor_find(&c, "resnet.seg_1.weight");
    REQUIRE(t && t->shape[0] == 256 && t->shape[1] == 5120);
    diar_checkpoint_close(&c);
    diar_plda *plda = malloc(sizeof(*plda));
    REQUIRE(plda);
    snprintf(path, sizeof(path), "%s/plda-plda.npz", dir);
    snprintf(other, sizeof(other), "%s/plda-xvec_transform.npz", dir);
    REQUIRE(!diar_plda_open(plda, other, path));
    float embedding[256];
    for (int i = 0; i < 256; ++i)
        embedding[i] = sinf(i);
    double out[128];
    diar_plda_apply(plda, embedding, out);
    for (int i = 0; i < 128; ++i)
        REQUIRE(isfinite(out[i]) && plda->phi[i] > 0);
    float e[4 * 256], cent[4 * 256];
    for (int i = 0; i < 4; ++i)
        for (int d = 0; d < 256; ++d)
            e[i * 256 + d] = embedding[d];
    REQUIRE(diar_cluster(plda, e, 4, cent) == 1);
    for (int d = 0; d < 256; ++d)
        REQUIRE(fabs(cent[d] - embedding[d]) < 1e-6);
    free(plda);
    return 0;
}
static int aba_output(const char *path) {
    FILE *f = fopen(path, "rb");
    REQUIRE(f);
    char text[65536];
    size_t n = fread(text, 1, sizeof(text) - 1, f);
    REQUIRE(!ferror(f) && feof(f));
    fclose(f);
    text[n] = 0;
    REQUIRE(strstr(text, "\"num_speakers\":2"));
    char *p = strstr(text, "\"diarization\":[");
    REQUIRE(p);
    p = strchr(p, '[') + 1;
    int a = -1, b = -1, returning = -1, count = 0;
    double previous = -1, coverage[3] = {0};
    while (*p && *p != ']') {
        double start, end;
        int speaker, used = 0;
        REQUIRE(sscanf(p, "{\"start\":%lf,\"end\":%lf,\"speaker\":\"SPEAKER_%d\"}%n", &start, &end,
                       &speaker, &used) == 3 &&
                used > 0);
        REQUIRE(isfinite(start) && isfinite(end) && start >= previous && end > start && end <= 26);
        previous = start;
        double mid = (start + end) / 2;
        if (mid < 8) {
            if (a < 0)
                a = speaker;
            REQUIRE(speaker == a);
            coverage[0] += end - start;
        } else if (mid >= 9 && mid < 17) {
            if (b < 0)
                b = speaker;
            REQUIRE(speaker == b);
            coverage[1] += end - start;
        } else if (mid >= 18) {
            if (returning < 0)
                returning = speaker;
            REQUIRE(speaker == returning);
            coverage[2] += end - start;
        } else
            REQUIRE(0);
        p += used;
        if (*p == ',')
            ++p;
        ++count;
    }
    REQUIRE(*p == ']' && count >= 3 && a >= 0 && b >= 0 && a != b && returning == a);
    for (int i = 0; i < 3; ++i)
        REQUIRE(coverage[i] > 3);
    puts("two-speaker A-B-A identity checks passed (not a DER evaluation)");
    return 0;
}
int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--check-aba"))
        return aba_output(argv[2]);
    REQUIRE(argc <= 2);
    REQUIRE(!kernels());
    REQUIRE(!clustering());
    REQUIRE(!invalid_archive());
    if (argc == 2)
        REQUIRE(!checkpoints(argv[1]));
    puts("diarization C checks passed");
    return 0;
}
