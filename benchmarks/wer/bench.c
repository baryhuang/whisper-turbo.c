#define _POSIX_C_SOURCE 200809L
#include "score.h"
#include <cjson/cJSON.h>
#include <curl/curl.h>
#include <sndfile.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

static void die(const char *s) { fprintf(stderr, "WER benchmark: %s\n", s); exit(1); }
static const char *str(const cJSON *o, const char *key) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, key);
    if (!cJSON_IsString(v)) die(key);
    return v->valuestring;
}
static double num(const cJSON *o, const char *key) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, key);
    if (!cJSON_IsNumber(v) || !isfinite(v->valuedouble)) die(key);
    return v->valuedouble;
}
static void path(char out[4096], const char *dir, const char *name, const char *suffix) {
    if (snprintf(out, 4096, "%s/%s%s", dir, name, suffix) >= 4096) die("path too long");
}
static int safe_id(const char *s) {
    if (!*s || strlen(s) > 80) return 0;
    for (; *s; ++s) if (!((*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'z') || *s == '-')) return 0;
    return 1;
}
typedef struct { char *data; size_t used; } buffer;
static size_t receive(char *data, size_t size, size_t count, void *opaque) {
    buffer *b = opaque;
    if (size && count > SIZE_MAX / size) return 0;
    size_t n = size * count;
    if (n > 16000000 - b->used) return 0;
    char *p = realloc(b->data, b->used + n + 1);
    if (!p) return 0;
    b->data = p; memcpy(p + b->used, data, n); b->used += n; p[b->used] = 0; return n;
}
static CURL *client(const char *url, buffer *b) {
    CURL *c = curl_easy_init(); if (!c) die("curl init");
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS_STR, "https");
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 900L);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, receive);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, b);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "whisper-turbo-c-wer/1");
    return c;
}
static buffer download(const char *url) {
    buffer b = {0}; CURL *c = client(url, &b);
    CURLcode rc = curl_easy_perform(c); long status = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status); curl_easy_cleanup(c);
    if (rc || status != 200) die("dataset download failed");
    return b;
}
static void write_bytes(const char *name, const void *data, size_t n) {
    FILE *f = fopen(name, "wx"); if (!f) die("cannot create output (already exists?)");
    if (fwrite(data, 1, n, f) != n || fclose(f)) die("output write failed");
}
static cJSON *read_json(const char *name) {
    FILE *f = fopen(name, "rb"); if (!f) die("cannot read JSON");
    if (fseek(f, 0, SEEK_END)) die("seek");
    long n = ftell(f); if (n < 0 || n > 16000000 || fseek(f, 0, SEEK_SET)) die("JSON size");
    char *p = calloc((size_t)n + 1, 1); if (!p) die("allocation");
    if (fread(p, 1, (size_t)n, f) != (size_t)n || fclose(f)) die("read");
    cJSON *j = cJSON_ParseWithLengthOpts(p, (size_t)n + 1, NULL, 1); free(p);
    if (!j) die("invalid JSON");
    return j;
}
static void write_json(const char *name, cJSON *j) {
    char *s = cJSON_Print(j); if (!s) die("JSON serialization");
    write_bytes(name, s, strlen(s)); free(s);
}
static void hash_file(const char *name, char hex[65]) {
    FILE *f = fopen(name, "rb"); EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!f || !ctx || !EVP_DigestInit_ex(ctx, EVP_sha256(), NULL)) die("hash init");
    unsigned char block[8192], digest[32]; size_t n; unsigned len;
    for (;;) {
        n = fread(block, 1, sizeof(block), f);
        if (ferror(f)) die("hash read");
        if (n && !EVP_DigestUpdate(ctx, block, n)) die("hash update");
        if (feof(f)) break;
    }
    if (fclose(f) || !EVP_DigestFinal_ex(ctx, digest, &len) || len != 32) die("hash read");
    EVP_MD_CTX_free(ctx);
    for (size_t i = 0; i < 32; ++i) snprintf(hex + 2*i, 3, "%02x", digest[i]);
}
static void prepare(const char *dir, unsigned count, unsigned offset) {
    if (!count || count > 2620 || offset > 2620 - count) die("invalid dataset range");
    if (mkdir(dir, 0700)) die("dataset directory must not exist");
    buffer info = download("https://huggingface.co/api/datasets/openslr/librispeech_asr");
    cJSON *meta = cJSON_Parse(info.data); if (!meta) die("dataset metadata");
    char revision[64]; snprintf(revision, sizeof(revision), "%s", str(meta, "sha"));
    free(info.data); cJSON_Delete(meta);
    cJSON *manifest = cJSON_CreateObject(), *items = cJSON_AddArrayToObject(manifest, "items");
    cJSON_AddStringToObject(manifest, "dataset", "openslr/librispeech_asr");
    cJSON_AddStringToObject(manifest, "config", "clean");
    cJSON_AddStringToObject(manifest, "split", "test");
    cJSON_AddStringToObject(manifest, "dataset_revision", revision);
    cJSON_AddStringToObject(manifest, "license", "CC-BY-4.0");
    cJSON_AddStringToObject(manifest, "selection", "Consecutive Hugging Face viewer rows; no filtering by duration or output.");
    cJSON_AddNumberToObject(manifest, "offset", offset);
    cJSON_AddNumberToObject(manifest, "count", count);
    for (unsigned start = 0; start < count; start += 100) {
        unsigned length = count - start < 100 ? count - start : 100;
        char url[512]; snprintf(url, sizeof(url), "https://datasets-server.huggingface.co/rows?dataset=openslr%%2Flibrispeech_asr&config=clean&split=test&offset=%u&length=%u", offset + start, length);
        buffer b = download(url); cJSON *page = cJSON_Parse(b.data); free(b.data);
        cJSON *rows = cJSON_GetObjectItemCaseSensitive(page, "rows");
        if (cJSON_GetArraySize(rows) != (int)length) die("missing dataset rows");
        for (unsigned i = 0; i < length; ++i) {
            cJSON *entry = cJSON_GetArrayItem(rows, (int)i), *row = cJSON_GetObjectItemCaseSensitive(entry, "row");
            const char *id = str(row, "id"), *text = str(row, "text");
            if (!safe_id(id) || num(entry, "row_idx") != offset + start + i) die("dataset row identity");
            cJSON *audio = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(row, "audio"), 0);
            const char *src = str(audio, "src");
            if (strncmp(src, "https://datasets-server.huggingface.co/", 39) || !strstr(src, revision)) die("dataset revision changed");
            buffer flac = download(src); char file[4096], wav[4096], hash[65];
            path(file, dir, id, ".flac"); path(wav, dir, id, ".wav");
            write_bytes(file, flac.data, flac.used); free(flac.data);
            SF_INFO sf = {0}; SNDFILE *input = sf_open(file, SFM_READ, &sf);
            if (!input || sf.samplerate != 16000 || sf.channels != 1 || sf.frames <= 0 || sf.frames > 4800000) die("unexpected corpus audio");
            sf_count_t frames = sf.frames;
            SF_INFO wf = {.samplerate=16000, .channels=1, .format=SF_FORMAT_WAV | SF_FORMAT_PCM_16};
            SNDFILE *output = sf_open(wav, SFM_WRITE, &wf); if (!output) die("WAV output");
            short samples[8192]; sf_count_t got, total = 0;
            while ((got = sf_read_short(input, samples, 8192)) > 0) {
                if (sf_write_short(output, samples, got) != got) die("WAV write");
                total += got;
            }
            if (total != frames || sf_close(input) || sf_close(output)) die("FLAC decode");
            cJSON *item = cJSON_CreateObject(); cJSON_AddItemToArray(items, item);
            cJSON_AddStringToObject(item, "id", id); cJSON_AddStringToObject(item, "reference", text);
            cJSON_AddNumberToObject(item, "row", offset + start + i);
            cJSON_AddNumberToObject(item, "speaker", num(row, "speaker_id"));
            cJSON_AddNumberToObject(item, "samples", (double)frames);
            cJSON_AddNumberToObject(item, "duration", frames / 16000.0);
            hash_file(file, hash); cJSON_AddStringToObject(item, "flac_sha256", hash);
            hash_file(wav, hash); cJSON_AddStringToObject(item, "wav_sha256", hash);
            fprintf(stderr, "prepared %u/%u %s\n", start+i+1, count, id);
        }
        cJSON_Delete(page);
    }
    char file[4096]; path(file, dir, "manifest", ".json"); write_json(file, manifest); cJSON_Delete(manifest);
}
static void field(curl_mime *form, const char *name, const char *value) {
    curl_mimepart *p = curl_mime_addpart(form);
    if (!p || curl_mime_name(p, name) || curl_mime_data(p, value, CURL_ZERO_TERMINATED)) die("multipart field");
}
static cJSON *request(const char *url, const char *wav, int cpp, double *elapsed) {
    buffer b = {0}; CURL *c = client(url, &b); curl_mime *form = curl_mime_init(c);
    curl_mimepart *p = curl_mime_addpart(form);
    if (!form || !p || curl_mime_name(p, "file") || curl_mime_filedata(p, wav)) die("multipart audio");
    field(form, "model", "whisper-1"); field(form, "language", "en");
    field(form, "response_format", "json"); field(form, "temperature", "0");
    if (cpp) { field(form, "beam_size", "5"); field(form, "best_of", "5"); field(form, "temperature_inc", "0.2"); }
    curl_easy_setopt(c, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(c, CURLOPT_MIMEPOST, form);
    CURLcode rc = curl_easy_perform(c); long status = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status); curl_easy_getinfo(c, CURLINFO_TOTAL_TIME, elapsed);
    curl_mime_free(form); curl_easy_cleanup(c);
    cJSON *result = cJSON_CreateObject();
    cJSON_AddNumberToObject(result, "http_status", status); cJSON_AddNumberToObject(result, "curl_code", rc);
    cJSON *body = b.data ? cJSON_ParseWithLengthOpts(b.data, b.used + 1, NULL, 1) : NULL;
    const cJSON *text = cJSON_GetObjectItemCaseSensitive(body, "text");
    int ok = !rc && status == 200 && cJSON_IsString(text);
    cJSON_AddBoolToObject(result, "success", ok);
    cJSON_AddStringToObject(result, "hypothesis", ok ? text->valuestring : "");
    if (!ok && b.data) cJSON_AddStringToObject(result, "error_body", b.data);
    free(b.data); cJSON_Delete(body); return result;
}
static void run(const char *dir, const char *url, const char *label, int cpp) {
    if (!safe_id(label) || strncmp(url, "http://127.0.0.1:", 17)) die("label or non-loopback URL");
    const char *port = url + 17; unsigned p = 0;
    if (*port < '0' || *port > '9') die("invalid loopback port");
    for (; *port >= '0' && *port <= '9'; ++port) {
        p = p * 10 + (unsigned)(*port - '0'); if (p > 65535) die("invalid loopback port");
    }
    if (!p || *port != '/') die("invalid loopback endpoint");
    char file[4096]; path(file, dir, "manifest", ".json"); cJSON *manifest = read_json(file);
    cJSON *items = cJSON_GetObjectItemCaseSensitive(manifest, "items"); int n = cJSON_GetArraySize(items);
    if (n < 1) die("empty manifest");
    path(file, dir, label, ".jsonl"); FILE *f = fopen(file, "wx"); if (!f) die("results already exist");
    for (int i = -1; i < n; ++i) {
        cJSON *item = cJSON_GetArrayItem(items, i < 0 ? 0 : i); const char *id = str(item, "id");
        if (!safe_id(id)) die("invalid manifest id");
        char wav[4096], hash[65]; path(wav, dir, id, ".wav"); hash_file(wav, hash);
        if (strcmp(hash, str(item, "wav_sha256"))) die("audio hash mismatch");
        double elapsed; cJSON *result = request(url, wav, cpp, &elapsed);
        cJSON_AddStringToObject(result, "id", id); cJSON_AddNumberToObject(result, "seconds", elapsed);
        if (i < 0) {
            path(file, dir, label, "-warmup.json"); write_json(file, result);
            if (!cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(result, "success"))) die("warmup failed");
        } else {
            cJSON_AddStringToObject(result, "reference", str(item, "reference"));
            cJSON_AddNumberToObject(result, "audio_seconds", num(item, "duration"));
            char *s = cJSON_PrintUnformatted(result); if (!s || fprintf(f, "%s\n", s) < 0 || fflush(f)) die("results write");
            free(s);
            fprintf(stderr, "%s %d/%d %s HTTP %.0f %.3fs\n", label, i+1, n, id, num(result, "http_status"), elapsed);
        }
        cJSON_Delete(result);
    }
    if (fclose(f)) die("results close");
    cJSON_Delete(manifest);
}
static void score(const char *dir, const char *label) {
    if (!safe_id(label)) die("invalid label");
    char file[4096]; path(file, dir, "manifest", ".json"); cJSON *manifest = read_json(file);
    cJSON *items = cJSON_GetObjectItemCaseSensitive(manifest, "items"); int n = cJSON_GetArraySize(items);
    path(file, dir, label, ".jsonl"); FILE *f = fopen(file, "r"); if (!f) die("missing results");
    cJSON *summary = cJSON_CreateObject(), *per = cJSON_AddArrayToObject(summary, "utterances");
    char *line = NULL; size_t cap = 0; int index = 0, failures = 0;
    wer_counts totals = {0}; double seconds = 0, duration = 0;
    while (getline(&line, &cap, f) >= 0) {
        cJSON *result = cJSON_ParseWithLengthOpts(line, strlen(line) + 1, NULL, 1), *item = cJSON_GetArrayItem(items, index++);
        if (!result || !item || strcmp(str(item, "id"), str(result, "id")) || strcmp(str(item, "reference"), str(result, "reference"))) die("result/manifest mismatch");
        int ok = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(result, "success")); failures += !ok;
        wer_counts c; if (wer_score(str(item, "reference"), ok ? str(result, "hypothesis") : "", &c) || !c.reference) die("scoring failed");
        totals.reference += c.reference; totals.hypothesis += c.hypothesis;
        totals.substitutions += c.substitutions; totals.deletions += c.deletions; totals.insertions += c.insertions;
        seconds += num(result, "seconds"); duration += num(item, "duration");
        cJSON *entry = cJSON_CreateObject(); cJSON_AddItemToArray(per, entry);
        cJSON_AddStringToObject(entry, "id", str(item, "id"));
        cJSON_AddNumberToObject(entry, "reference_words", c.reference);
        cJSON_AddNumberToObject(entry, "substitutions", c.substitutions);
        cJSON_AddNumberToObject(entry, "deletions", c.deletions);
        cJSON_AddNumberToObject(entry, "insertions", c.insertions);
        cJSON_Delete(result);
    }
    if (ferror(f) || fclose(f) || index != n) die("incomplete results; not a completed WER run");
    free(line);
    cJSON_AddStringToObject(summary, "engine", label);
    cJSON_AddStringToObject(summary, "normalization", "Unicode lowercase + NFKD; retain letters/numbers/underscore; collapse whitespace; no number or contraction expansion");
    cJSON_AddNumberToObject(summary, "utterance_count", index); cJSON_AddNumberToObject(summary, "failed_requests", failures);
    cJSON_AddNumberToObject(summary, "reference_words", totals.reference);
    cJSON_AddNumberToObject(summary, "substitutions", totals.substitutions);
    cJSON_AddNumberToObject(summary, "deletions", totals.deletions);
    cJSON_AddNumberToObject(summary, "insertions", totals.insertions);
    cJSON_AddNumberToObject(summary, "wer_percent", 100.0 * (totals.substitutions + totals.deletions + totals.insertions) / totals.reference);
    cJSON_AddNumberToObject(summary, "request_seconds", seconds); cJSON_AddNumberToObject(summary, "audio_seconds", duration);
    path(file, dir, label, "-summary.json"); write_json(file, summary);
    printf("%s: %.4f%% WER; %d/%d requests failed; %.3fs request time, %.3fs audio\n", label, num(summary, "wer_percent"), failures, n, seconds, duration);
    cJSON_Delete(summary); cJSON_Delete(manifest);
}
int main(int argc, char **argv) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT)) die("curl global init");
    if (argc >= 3 && !strcmp(argv[1], "prepare")) {
        char *end = NULL; unsigned long n = argc > 3 ? strtoul(argv[3], &end, 10) : 50;
        if ((end && *end) || !n || n > 2620) die("invalid count");
        unsigned long offset = argc > 4 ? strtoul(argv[4], &end, 10) : 0;
        if ((end && *end) || offset > 2620) die("invalid offset");
        prepare(argv[2], (unsigned)n, (unsigned)offset);
    } else if (argc == 6 && !strcmp(argv[1], "run")) {
        if (strcmp(argv[5], "native") && strcmp(argv[5], "cpp")) die("engine must be native or cpp");
        run(argv[2], argv[3], argv[4], !strcmp(argv[5], "cpp"));
    } else if (argc == 4 && !strcmp(argv[1], "score")) score(argv[2], argv[3]);
    else die("usage: wer-bench prepare DIR [COUNT [OFFSET]] | run DIR LOOPBACK_URL LABEL native|cpp | score DIR LABEL");
    curl_global_cleanup(); return 0;
}
