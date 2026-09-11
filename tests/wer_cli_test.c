#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cjson/cJSON.h>
static void write_fixture(const char *dir, const char *name, const char *text) {
    char p[512]; assert(snprintf(p, sizeof(p), "%s/%s", dir, name) < (int)sizeof(p));
    FILE *f = fopen(p, "w"); assert(f); assert(fputs(text, f) >= 0); assert(!fclose(f));
}
static int score(const char *exe, const char *dir, const char *label) {
    pid_t p = fork(); assert(p >= 0);
    if (!p) {
        int fd = open("/dev/null", O_WRONLY); if (fd < 0) _exit(126);
        if (dup2(fd, 1) < 0 || dup2(fd, 2) < 0) _exit(126);
        close(fd); execl(exe, exe, "score", dir, label, (char *)NULL); _exit(127);
    }
    int status; assert(waitpid(p, &status, 0) == p);
    assert(WIFEXITED(status)); return WEXITSTATUS(status);
}
static cJSON *read_summary(const char *dir) {
    char p[512], data[8192]; snprintf(p, sizeof(p), "%s/pass-summary.json", dir);
    FILE *f = fopen(p, "r"); assert(f);
    size_t n = fread(data, 1, sizeof(data)-1, f); assert(!ferror(f) && feof(f));
    data[n] = 0; fclose(f); cJSON *j = cJSON_Parse(data); assert(j); return j;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    char dir[] = "/tmp/wer-cli-test.XXXXXX"; assert(mkdtemp(dir));
    write_fixture(dir, "manifest.json", "{\"items\":[{\"id\":\"a\",\"reference\":\"one two\",\"duration\":1},{\"id\":\"b\",\"reference\":\"three four\",\"duration\":2}]}");
    const char *first = "{\"id\":\"a\",\"reference\":\"one two\",\"hypothesis\":\"one two\",\"success\":true,\"seconds\":0.1}\n";
    const char *failed = "{\"id\":\"b\",\"reference\":\"three four\",\"hypothesis\":\"this must be ignored\",\"success\":false,\"seconds\":0.2}\n";
    char both[1024]; snprintf(both, sizeof(both), "%s%s", first, failed);
    write_fixture(dir, "pass.jsonl", both);
    assert(score(argv[1], dir, "pass") == 0);
    cJSON *j = read_summary(dir);
    assert(cJSON_GetObjectItemCaseSensitive(j, "wer_percent")->valuedouble == 50);
    assert(cJSON_GetObjectItemCaseSensitive(j, "reference_words")->valueint == 4);
    assert(cJSON_GetObjectItemCaseSensitive(j, "deletions")->valueint == 2);
    assert(cJSON_GetObjectItemCaseSensitive(j, "failed_requests")->valueint == 1);
    cJSON_Delete(j);
    assert(score(argv[1], dir, "pass") == 1); /* Never overwrite a result. */
    write_fixture(dir, "short.jsonl", first); assert(score(argv[1], dir, "short") == 1);
    snprintf(both, sizeof(both), "%s%s", failed, first);
    write_fixture(dir, "wrong-order.jsonl", both); assert(score(argv[1], dir, "wrong-order") == 1);
    snprintf(both, sizeof(both), "%s%s", first, first);
    write_fixture(dir, "duplicate.jsonl", both); assert(score(argv[1], dir, "duplicate") == 1);
    write_fixture(dir, "malformed.jsonl", "not json\n"); assert(score(argv[1], dir, "malformed") == 1);
    const char *names[] = {"manifest.json", "pass.jsonl", "pass-summary.json", "short.jsonl", "wrong-order.jsonl", "duplicate.jsonl", "malformed.jsonl"};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
        char p[512]; snprintf(p, sizeof(p), "%s/%s", dir, names[i]); assert(!unlink(p));
    }
    assert(!rmdir(dir));
    puts("C WER CLI tests passed: failed requests count as deletions; incomplete, reordered, duplicate and malformed results rejected; no overwrite.");
}
