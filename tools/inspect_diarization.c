#include "checkpoint.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: inspect-diarization CHECKPOINT.bin\n");
        return 2;
    }
    diar_checkpoint c;
    if (diar_checkpoint_open(&c, argv[1]))
        return 1;
    for (size_t i = 0; i < c.count; ++i) {
        const diar_tensor *t = &c.tensors[i];
        printf("%s %s", t->name, t->dtype == 4 ? "f32" : "i64");
        for (int d = 0; d < t->rank; ++d)
            printf(" %zu", t->shape[d]);
        putchar('\n');
    }
    diar_checkpoint_close(&c);
    return 0;
}
