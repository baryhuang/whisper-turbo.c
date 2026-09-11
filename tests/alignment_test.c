#include "alignment.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* A final token with a zero-duration DTW span must not discard the whole
       recording. Preserve it in the preceding aligned group, with unchanged time. */
    const unsigned char tail[] = "Hello there !";
    size_t offsets[] = {0, 5, 11, 13}, collapsed[] = {0, 10, 20, 20};
    wt_word grouped[4]; size_t grouped_count = 0;
    assert(!wt_alignment_words(tail, 13, offsets, collapsed, 3, 0, 16000,
                                grouped, 4, &grouped_count));
    assert(grouped_count == 2 && grouped[1].offset == 5 && grouped[1].length == 8 &&
           grouped[1].start == .2 && grouped[1].end == .4);
    /* Do not invent time for an entirely collapsed window or borrow a span
       from the previous ASR window. */
    size_t all_zero[] = {0,0,0,0};
    assert(wt_alignment_words(tail,13,offsets,all_zero,3,480000,960000,
                               grouped,4,&grouped_count));
    size_t descending[] = {0,10,9,20}; grouped_count = 0;
    assert(wt_alignment_words(tail,13,offsets,descending,3,0,16000,grouped,4,&grouped_count));
    assert(wt_alignment_words(tail,13,offsets,collapsed,3,0,16000,grouped,1,&grouped_count));
    /* Final window of a five-minute request uses absolute, bounded timestamps. */
    grouped_count = 0;
    assert(!wt_alignment_words(tail,13,offsets,collapsed,3,4320000,4800000,
                                grouped,4,&grouped_count));
    assert(grouped[0].start == 270 && grouped[1].end == 270.4);
    unsigned char many_text[4000]; size_t many_offsets[201], many_bounds[201];
    wt_word *many_words=calloc(2000,sizeof(wt_word)); assert(many_words);
    for(size_t i=0;i<2000;++i) {many_text[2*i]=' ';many_text[2*i+1]='a';}
    size_t many_count=0;
    for(size_t window=0;window<10;++window) {
        for(size_t i=0;i<=200;++i) {many_offsets[i]=window*400+i*2;many_bounds[i]=i;}
        assert(!wt_alignment_words(many_text,sizeof(many_text),many_offsets,many_bounds,
                                    200,window*480000,4800000,many_words,2000,&many_count));
    }
    assert(many_count==2000 && many_words[1999].offset+many_words[1999].length==4000);
    free(many_words);
    float q[6 * 8 * 30];
    size_t bounds[4];
    for (size_t h = 0; h < 6; ++h)
        for (size_t t = 0; t < 8; ++t)
            for (size_t f = 0; f < 30; ++f) {
                double center = t < 3 || t == 7 ? 15 : (t - 3) * 9;
                double dx = f - center;
                q[(h * 8 + t) * 30 + f] = (float)(-dx * dx / 12);
            }
    assert(!wt_align(q, 8, 3, 30, 30, 8, bounds));
    assert(bounds[0] == 0 && bounds[1] > 0 && bounds[2] > bounds[1] && bounds[3] > bounds[2] && bounds[3] < 30);
    assert(wt_align(q, 7, 3, 30, 30, 8, bounds));
    q[0] = NAN;
    assert(wt_align(q, 8, 3, 30, 30, 8, bounds));
    wt_result r = {.duration=26, .length=14, .word_count=3};
    r.text = malloc(15); assert(r.text); memcpy(r.text, "Hello Hi Again", 15);
    r.words = calloc(3, sizeof(wt_word)); assert(r.words);
    r.words[0] = (wt_word){0,5,1,2};
    r.words[1] = (wt_word){5,3,11,12};
    r.words[2] = (wt_word){8,6,21,22};
    diar_interval turns[] = {{0,8,0},{9,17,1},{18,26,0}};
    diar_result d = {.exclusive=turns,.exclusive_count=3,.activity=turns,.activity_count=3,.speakers=2};
    assert(wt_speech_window_active(NULL, 0, 480000));
    assert(wt_speech_window_active(&d, 0, 480000));
    assert(!wt_speech_window_active(&d, 480000, 480000));
    diar_result quiet = {0};
    assert(!wt_speech_window_active(&quiet, 0, 480000));
    diar_interval edge[] = {{29.9, 30.1, 0}};
    diar_result boundary = {.activity=edge, .activity_count=1};
    assert(wt_speech_window_active(&boundary, 0, 480000));
    assert(wt_speech_window_active(&boundary, 480000, 480000));
    assert(!wt_speech_window_active(&boundary, 960000, 480000));
    assert(wt_language_window_offset(NULL, 4800000) == 0);
    assert(wt_language_window_offset(&quiet, 4800000) == 0);
    assert(wt_language_window_offset(&d, 480000) == 0);
    diar_interval language_activity[] = {{0.5, 1.0, 0}, {151, 174, 0}, {181, 195, 0}};
    diar_result language_speech = {.activity=language_activity, .activity_count=3};
    assert(wt_language_window_offset(&language_speech, 4800000) == 5 * 480000);
    assert(wt_speech_window_coverage(&language_speech, 0, 480000) == 0.5);
    assert(wt_speech_window_coverage(&language_speech, 5 * 480000, 480000) == 23);
    assert(wt_speech_window_coverage(&language_speech, 5 * 480000, 2 * 16000) == 1);
    assert(wt_speech_window_coverage(NULL, 0, 480000) == 0);
    /* A partial final window is clipped to actual samples; no padded duration. */
    assert(wt_language_window_offset(&language_speech, 152 * 16000) == 5 * 480000);
    diar_interval tied[] = {{1, 10, 0}, {31, 40, 0}};
    diar_result tie = {.activity=tied, .activity_count=2};
    assert(wt_language_window_offset(&tie, 960000) == 0);
    char names[32][64] = {{0}}; strcpy(names[0], "agent");
    wt_request req = {.diarize=1,.diarized_json=1,.stream=1}; wt_error e = {0};
    assert(!wt_assign_speakers(&r,&d,(const char (*)[64])names,&req,&e));
    assert(r.segment_count==3 && !strcmp(r.segments[0].speaker,"agent") &&
           !strcmp(r.segments[1].speaker,"A") && !strcmp(r.segments[2].speaker,"agent"));
    size_t offset=0;
    for(size_t i=0;i<r.segment_count;++i) {
        assert(!memcmp(r.text+offset,r.segments[i].text,r.segments[i].length));
        offset+=r.segments[i].length;
    }
    assert(offset==r.length && !strcmp((char *)r.text,"Hello Hi Again"));
    char *out=NULL; size_t n; const char *type;
    assert(!wt_render(&req,&r,&out,&n,&type));
    assert(strstr(out,"\"delta\":\" Hi\"") && strstr(out,"\"delta\":\" Again\""));
    free(out); wt_result_free(&r);
    wt_result gated = {.duration=300, .length=14, .word_count=3};
    gated.text=malloc(15); assert(gated.text); memcpy(gated.text,"Hello Hi Again",15);
    gated.words=calloc(3,sizeof(wt_word)); assert(gated.words);
    gated.words[0]=(wt_word){0,5,1,2};
    gated.words[1]=(wt_word){5,3,119.8,119.88};
    gated.words[2]=(wt_word){8,6,210,211};
    diar_interval supported[]={{0,3,0},{209,212,0}};
    diar_result support={.activity=supported,.activity_count=2};
    assert(!wt_filter_speech_words(&gated,&support,&e));
    /* Internal gaps cannot justify deleting distant/weak speech. */
    assert(gated.duration==300 && gated.word_count==3 && gated.length==14);
    assert(!strcmp((char *)gated.text,"Hello Hi Again") && gated.words[2].offset==8);
    assert(gated.words[2].start==210 && gated.words[2].end==211);
    assert(!wt_filter_speech_words(&gated,&quiet,&e));
    assert(!gated.word_count && !gated.length && !gated.text[0] && gated.duration==300);
    wt_result_free(&gated);
    /* Mere overlap does not validate a group stretching 27 s into the final
       silence. Entirely unsupported trailing groups are removed too. */
    wt_result stretched = {.duration=300, .length=14, .word_count=3};
    stretched.text=malloc(15); assert(stretched.text); memcpy(stretched.text,"Hello Hi Again",15);
    stretched.words=calloc(3,sizeof(wt_word)); assert(stretched.words);
    stretched.words[0]=(wt_word){0,5,0,10};
    stretched.words[1]=(wt_word){5,3,90,119.8};
    stretched.words[2]=(wt_word){8,6,210,211};
    diar_interval speech_spans[]={{0,3,0},{2.9,6,0},{90,92.86,0}};
    diar_result speech_union={.activity=speech_spans,.activity_count=3};
    assert(!wt_filter_speech_words(&stretched,&speech_union,&e));
    assert(stretched.word_count==1 && !strcmp((char *)stretched.text,"Hello"));
    assert(stretched.words[0].end==10 && stretched.duration==300);
    diar_interval duplicate_spans[]={{0,3,0},{0,3,0}};
    diar_result duplicate_support={.activity=duplicate_spans,.activity_count=2};
    assert(!wt_filter_speech_words(&stretched,&duplicate_support,&e));
    assert(!stretched.word_count && !stretched.length);
    wt_result_free(&stretched);
    puts("C alignment tests passed: DTW, bounds, A-B-A labels, unchanged text, SSE whitespace.");
    return 0;
}
