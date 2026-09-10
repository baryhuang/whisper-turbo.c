#include "whisper_turbo_q8.h"
#define WHISPER_TURBO_HAVE_Q8_GEMM 1
#define whisper_turbo_q8_gemm wt_q8_gemm_auto
#include "whisper_turbo_attention.h"
#define WHISPER_TURBO_HAVE_SELF_ATTENTION 1
#define whisper_turbo_self_attention wt_attention
