#include "../benchmarks/wer/score.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t independent_distance(const unsigned *a, size_t n, const unsigned *b, size_t m) {
    size_t d[9][9];
    for (size_t i = 0; i <= n; ++i) d[i][0] = i;
    for (size_t j = 0; j <= m; ++j) d[0][j] = j;
    for (size_t i = 1; i <= n; ++i) for (size_t j = 1; j <= m; ++j) {
        size_t v = d[i-1][j-1] + (a[i-1] != b[j-1]);
        if (d[i-1][j] + 1 < v) v = d[i-1][j] + 1;
        if (d[i][j-1] + 1 < v) v = d[i][j-1] + 1;
        d[i][j] = v;
    }
    return d[n][m];
}
int main(void) {
    char *s = wer_normalize("  HELLO,\tCaf\xc3\xa9! Don't re-enter １２.\n");
    assert(s && !strcmp(s, "hello cafe dont reenter 12")); free(s);
    s = wer_normalize("!!!"); assert(s && !*s); free(s);
    assert(!wer_normalize("\xff"));
    wer_counts c;
    assert(!wer_score("a b c", "a x c y", &c));
    assert(c.reference == 3 && c.hypothesis == 4 && c.substitutions == 1 && c.insertions == 1 && !c.deletions);
    assert(!wer_score("a b c", "a c", &c)); assert(c.deletions == 1 && !c.substitutions && !c.insertions);
    assert(!wer_score("a b", "", &c)); assert(c.deletions == 2);
    assert(!wer_score("", "a b", &c)); assert(c.insertions == 2 && !c.reference);
    assert(!wer_score("", "", &c)); assert(!c.reference && !c.insertions);
    assert(!wer_score("A, B!", "a b", &c)); assert(!c.substitutions && !c.deletions && !c.insertions);
    s = wer_normalize("Straße İ ﬁ ¼ Ⅳ foo_bar a—b");
    assert(s && !strcmp(s, "straße i fi 14 iv foo_bar ab")); free(s);
    s = wer_normalize(" ΑΒΓ ПРИВЕТ \x1c x ");
    assert(s && !strcmp(s, "αβγ привет x")); free(s);
    /* Cross-check the linear-space count implementation against an independent
       full distance matrix on 1,000 deterministic synthetic pairs. */
    unsigned seed = 1;
    for (unsigned trial = 0; trial < 1000; ++trial) {
        unsigned a[8], b[8]; char at[17] = {0}, bt[17] = {0};
        size_t n = trial % 9, m = (trial / 9) % 9;
        for (size_t i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u; a[i] = seed % 3;
            at[2*i] = (char)('a' + a[i]); at[2*i+1] = ' ';
        }
        for (size_t i = 0; i < m; ++i) {
            seed = seed * 1664525u + 1013904223u; b[i] = seed % 3;
            bt[2*i] = (char)('a' + b[i]); bt[2*i+1] = ' ';
        }
        assert(!wer_score(at, bt, &c));
        assert(c.reference == n && c.hypothesis == m);
        assert(c.substitutions + c.deletions + c.insertions == independent_distance(a,n,b,m));
        assert(n + c.insertions == m + c.deletions);
    }
    puts("C WER tests passed: Unicode normalization, S/D/I, empty outputs, invalid UTF-8 and 1,000 independent alignments.");
}
