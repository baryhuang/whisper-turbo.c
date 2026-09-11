#ifndef WT_WER_SCORE_H
#define WT_WER_SCORE_H
#include <stddef.h>
typedef struct { size_t reference, hypothesis, substitutions, deletions, insertions; } wer_counts;
/* Allocates UTF-8: lowercase, NFKD, delete punctuation/marks, collapse whitespace.
   Deliberately not Whisper's EnglishTextNormalizer (no number/contraction rules). */
char *wer_normalize(const char *text);
/* Sentence-local unit-cost Levenshtein. Ties: match/substitute, delete, insert.
   Caller sums counts across utterances, never averages sentence WERs. */
int wer_score(const char *reference, const char *hypothesis, wer_counts *out);
#endif
