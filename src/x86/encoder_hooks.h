#include "whisper_turbo_q8.h"
#define WHISPER_TURBO_HAVE_Q8_GROUP_DOT 1
#define whisper_turbo_q8_group_dot wt_q8_group_auto
#include "whisper_turbo_attention.h"
#define WHISPER_TURBO_HAVE_SELF_ATTENTION 1
#define whisper_turbo_self_attention wt_attention
