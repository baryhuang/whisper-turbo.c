#define _POSIX_C_SOURCE 200809L
#include "api.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define CONNECTIONS 8
#define HEADERS 8192
typedef struct server server;
typedef struct {
    server *s;
    int fd, claimed, done;
    double deadline;
    wt_request request;
    wt_result result;
    wt_error error;
} connection;
struct server {
    wt_server_options options;
    wt_backend backend;
    void *context;
    atomic_int *stop;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    connection clients[CONNECTIONS];
    connection *job;
    int busy, active, worker_stop;
};
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int wait_fd(int fd, short events, double deadline) {
    for (;;) {
        double left = deadline - now();
        if (left <= 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        struct pollfd p = {fd, events, 0};
        int rc = poll(&p, 1, left > 1 ? 1000 : (int)(left * 1000) + 1);
        if (rc < 0 && errno == EINTR)
            continue;
        if (rc < 0)
            return -1;
        if (rc && (p.revents & events))
            return 0;
        if (rc && (p.revents & (POLLHUP | POLLERR | POLLNVAL)))
            return -1;
    }
}
static int send_all(int fd, const void *buffer, size_t n, double deadline) {
    const unsigned char *p = buffer;
    while (n) {
        if (wait_fd(fd, POLLOUT, deadline))
            return -1;
#ifdef MSG_NOSIGNAL
        ssize_t sent = send(fd, p, n, MSG_NOSIGNAL);
#else
        ssize_t sent = send(fd, p, n, 0);
#endif
        if (sent < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        if (sent <= 0)
            return -1;
        p += sent;
        n -= (size_t)sent;
    }
    return 0;
}
static const char *reason(int code) {
    switch (code) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 401:
        return "Unauthorized";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 408:
        return "Request Timeout";
    case 411:
        return "Length Required";
    case 413:
        return "Content Too Large";
    case 415:
        return "Unsupported Media Type";
    case 417:
        return "Expectation Failed";
    case 422:
        return "Unprocessable Content";
    case 429:
        return "Too Many Requests";
    case 431:
        return "Request Header Fields Too Large";
    case 501:
        return "Not Implemented";
    case 503:
        return "Service Unavailable";
    case 504:
        return "Gateway Timeout";
    default:
        return "Internal Server Error";
    }
}
static void reply(int fd, int status, const char *type, const void *body, size_t length) {
    char header[512];
    int n = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n%s\r\n",
        status, reason(status), type, length,
        status == 429 || status == 503 ? "Retry-After: 1\r\n"
        : status == 401                ? "WWW-Authenticate: Bearer\r\n"
        : status == 405                ? "Allow: POST\r\n"
                                       : "");
    double deadline = now() + 5;
    if (n > 0 && (size_t)n < sizeof(header) && !send_all(fd, header, (size_t)n, deadline))
        (void)send_all(fd, body, length, deadline);
}
static void error_reply(int fd, wt_error e) {
    char *message = wt_json_string((const unsigned char *)e.message, strlen(e.message));
    char *param = e.param ? wt_json_string((const unsigned char *)e.param, strlen(e.param)) : NULL;
    char buffer[2048];
    int n = snprintf(buffer, sizeof(buffer),
                     "{\"error\":{\"message\":%s,\"type\":\"%s\",\"param\":%s,\"code\":\"%s\"}}",
                     message ? message : "\"Request failed.\"",
                     e.status == 401   ? "authentication_error"
                     : e.status == 429 ? "rate_limit_error"
                     : e.status >= 500 ? "server_error"
                                       : "invalid_request_error",
                     param ? param : "null", e.code);
    if (n > 0 && (size_t)n < sizeof(buffer))
        reply(fd, e.status, "application/json", buffer, (size_t)n);
    free(message);
    free(param);
}
static int cancelled(void *opaque) {
    connection *c = opaque;
    if (*c->s->stop || now() >= c->deadline)
        return 1;
    char byte;
    ssize_t n = recv(c->fd, &byte, 1, MSG_PEEK); /* Accepted sockets are nonblocking. */
    return n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
}
static void *worker(void *opaque) {
    server *s = opaque;
    pthread_mutex_lock(&s->mutex);
    for (;;) {
        while (!s->job && !s->worker_stop)
            pthread_cond_wait(&s->changed, &s->mutex);
        if (s->worker_stop && !s->job)
            break;
        connection *c = s->job;
        pthread_mutex_unlock(&s->mutex);
        int rc = s->backend(s->context, &c->request, &c->result, &c->error, cancelled, c);
        if (rc && !c->error.status)
            wt_fail(&c->error, 500, "Transcription failed.", NULL, "inference_error");
        if (!rc && (c->result.length > WT_TEXT_LIMIT || !c->result.text))
            wt_fail(&c->error, 500, "Invalid transcription output.", NULL, "inference_error");
        pthread_mutex_lock(&s->mutex);
        s->job = NULL;
        c->done = 1;
        pthread_cond_broadcast(&s->changed);
    }
    pthread_mutex_unlock(&s->mutex);
    return NULL;
}
static int authorized(const char *value, const char *key) {
    if (!key || !*key)
        return 1;
    if (strncasecmp(value, "Bearer ", 7))
        return 0;
    value += 7;
    size_t n = strlen(key);
    if (strlen(value) != n)
        return 0;
    unsigned diff = 0;
    for (size_t i = 0; i < n; ++i)
        diff |= (unsigned char)key[i] ^ (unsigned char)value[i];
    return diff == 0;
}
static int header_name(const char *s) {
    if (!*s)
        return 0;
    for (; *s; ++s)
        if (!isalnum((unsigned char)*s) && !strchr("!#$%&'*+-.^_`|~", *s))
            return 0;
    return 1;
}
static void *handle(void *opaque) {
    connection *c = opaque;
    server *s = c->s;
    char headers[HEADERS + 1], boundary[71], *type = NULL, *auth = "", *expect = NULL;
    size_t have = 0, content_length = 0, body_start = 0;
    unsigned seen = 0;
    unsigned char *body = NULL;
    wt_error e = {400, "Malformed HTTP request.", NULL, "invalid_request"};
    double deadline = now() + 10;
    for (;;) {
        if (have == HEADERS) {
            e.status = 431;
            e.message = "Request headers exceed 8192 bytes.";
            goto fail;
        }
        if (wait_fd(c->fd, POLLIN, deadline))
            goto timeout;
        ssize_t n = recv(c->fd, headers + have, HEADERS - have, 0);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
            continue;
        if (n <= 0)
            goto cleanup;
        have += (size_t)n;
        headers[have] = 0;
        char *end = strstr(headers, "\r\n\r\n");
        if (end) {
            body_start = (size_t)(end - headers) + 4;
            *end = 0;
            break;
        }
        if (memchr(headers, 0, have))
            goto fail;
    }
    /* Bare CR/LF, controls and obsolete folded headers are rejected. */
    for (size_t i = 0; i + 4 < body_start; ++i) {
        unsigned char ch = (unsigned char)headers[i];
        if (ch == '\r') {
            if (headers[i + 1] != '\n')
                goto fail;
            ++i;
        } else if (ch < 32 && ch != '\t')
            goto fail;
        else if (ch == 127)
            goto fail;
    }
    char *line = strstr(headers, "\r\n");
    if (!line)
        goto fail;
    *line = 0;
    char method[16], path[256], version[16], extra;
    if (sscanf(headers, "%15s %255s %15s %c", method, path, version, &extra) != 3 ||
        (strcmp(version, "HTTP/1.1") && strcmp(version, "HTTP/1.0")))
        goto fail;
    line += 2;
    while (*line) {
        char *next = strstr(line, "\r\n");
        if (next)
            *next = 0;
        char *colon = strchr(line, ':');
        if (!colon)
            goto fail;
        *colon++ = 0;
        if (!header_name(line))
            goto fail;
        while (*colon == ' ' || *colon == '\t')
            ++colon;
        char *tail = colon + strlen(colon);
        while (tail > colon && (tail[-1] == ' ' || tail[-1] == '\t'))
            *--tail = 0;
        unsigned bit = !strcasecmp(line, "Content-Length")      ? 1
                       : !strcasecmp(line, "Content-Type")      ? 2
                       : !strcasecmp(line, "Authorization")     ? 4
                       : !strcasecmp(line, "Expect")            ? 8
                       : !strcasecmp(line, "Host")              ? 16
                       : !strcasecmp(line, "Transfer-Encoding") ? 32
                                                                : 0;
        if (bit && (seen & bit))
            goto fail;
        seen |= bit;
        if (bit == 1) {
            if (!*colon)
                goto fail;
            for (const char *p = colon; *p; ++p) {
                if (*p < '0' || *p > '9')
                    goto fail;
                if (content_length > WT_UPLOAD_LIMIT / 10U)
                    goto too_large;
                content_length = content_length * 10 + (unsigned)(*p - '0');
                if (content_length > WT_UPLOAD_LIMIT)
                    goto too_large;
            }
        } else if (bit == 2)
            type = colon;
        else if (bit == 4)
            auth = colon;
        else if (bit == 8)
            expect = colon;
        else if (bit == 16 && !*colon)
            goto fail;
        if (!next)
            break;
        line = next + 2;
    }
    if (!strcmp(version, "HTTP/1.1") && !(seen & 16))
        goto fail;
    if (seen & 32) {
        wt_fail(&e, (seen & 1) ? 400 : 501,
                "Send Content-Length; transfer encodings are not supported.", NULL,
                "unsupported_transfer_encoding");
        goto fail;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        if (content_length)
            goto fail;
        reply(c->fd, 200, "application/json", "{\"status\":\"ready\"}", 18);
        goto cleanup;
    }
    if (!authorized(auth, s->options.api_key)) {
        wt_fail(&e, 401, "Invalid or missing bearer API key.", NULL, "invalid_api_key");
        goto fail;
    }
    if (strcmp(path, "/v1/audio/transcriptions")) {
        wt_fail(&e, 404, "Endpoint not found.", NULL, "not_found");
        goto fail;
    }
    if (strcmp(method, "POST")) {
        wt_fail(&e, 405, "Use POST.", NULL, "method_not_allowed");
        goto fail;
    }
    if (!(seen & 1)) {
        wt_fail(&e, 411, "Content-Length is required.", NULL, "length_required");
        goto fail;
    }
    if (!type || wt_boundary(type, boundary)) {
        wt_fail(&e, 415, "Use multipart/form-data with a valid boundary.", NULL,
                "invalid_content_type");
        goto fail;
    }
    if (expect && strcasecmp(expect, "100-continue")) {
        wt_fail(&e, 417, "Unsupported Expect header.", NULL, "invalid_expectation");
        goto fail;
    }
    pthread_mutex_lock(&s->mutex);
    if (!s->busy) {
        s->busy = 1;
        c->claimed = 1;
    }
    pthread_mutex_unlock(&s->mutex);
    if (!c->claimed) {
        wt_fail(&e, 429, "One transcription is already active; retry later.", NULL, "server_busy");
        goto fail;
    }
    body = malloc(content_length ? content_length : 1);
    if (!body) {
        wt_fail(&e, 503, "Upload allocation failed.", NULL, "resource_exhausted");
        goto fail;
    }
    size_t received = have - body_start;
    if (received > content_length)
        goto fail;
    memcpy(body, headers + body_start, received);
    deadline = now() + s->options.upload_seconds;
    if (expect && send_all(c->fd, "HTTP/1.1 100 Continue\r\n\r\n", 25, deadline))
        goto cleanup;
    while (received < content_length) {
        if (wait_fd(c->fd, POLLIN, deadline))
            goto timeout;
        ssize_t n = recv(c->fd, body + received, content_length - received, 0);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
            continue;
        if (n <= 0)
            goto cleanup;
        received += (size_t)n;
    }
    if (wt_multipart(body, content_length, boundary, &c->request, &e))
        goto fail;
    const unsigned char *pcm;
    size_t samples;
    if (wt_wav(c->request.file, c->request.file_size, &pcm, &samples, &e))
        goto fail;
    c->deadline = now() + s->options.timeout_seconds;
    pthread_mutex_lock(&s->mutex);
    s->job = c;
    pthread_cond_broadcast(&s->changed);
    while (!c->done)
        pthread_cond_wait(&s->changed, &s->mutex);
    pthread_mutex_unlock(&s->mutex);
    if (c->error.status) {
        e = c->error;
        goto fail;
    }
    if (c->request.plain_text)
        reply(c->fd, 200, "text/plain; charset=utf-8", c->result.text, c->result.length);
    else {
        char *quoted = wt_json_string(c->result.text, c->result.length);
        if (!quoted) {
            wt_fail(&e, 503, "Response allocation failed.", NULL, "resource_exhausted");
            goto fail;
        }
        size_t length = strlen(quoted) + 10;
        char *json = malloc(length);
        if (!json) {
            free(quoted);
            wt_fail(&e, 503, "Response allocation failed.", NULL, "resource_exhausted");
            goto fail;
        }
        int n = snprintf(json, length, "{\"text\":%s}", quoted);
        reply(c->fd, 200, "application/json", json, (size_t)n);
        free(json);
        free(quoted);
    }
    goto cleanup;
too_large:
    wt_fail(&e, 413, "Upload exceeds 25000000 bytes.", "file", "upload_too_large");
    goto fail;
timeout:
    wt_fail(&e, 408, "Request upload timed out.", NULL, "request_timeout");
fail:
    error_reply(c->fd, e);
cleanup:
    free(body);
    free(c->result.text);
    pthread_mutex_lock(&s->mutex);
    close(c->fd);
    c->fd = -1;
    if (c->claimed)
        s->busy = 0;
    --s->active;
    pthread_cond_broadcast(&s->changed);
    pthread_mutex_unlock(&s->mutex);
    return NULL;
}
int wt_serve(const wt_server_options *options, wt_backend backend, void *context,
             atomic_int *stop) {
    server s = {.options = *options,
                .backend = backend,
                .context = context,
                .stop = stop,
                .mutex = PTHREAD_MUTEX_INITIALIZER,
                .changed = PTHREAD_COND_INITIALIZER};
    for (int i = 0; i < CONNECTIONS; ++i)
        s.clients[i].fd = -1;
    int listener = socket(AF_INET, SOCK_STREAM, 0), enabled = 1;
    if (listener < 0)
        return -1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    struct sockaddr_in address = {.sin_family = AF_INET,
                                  .sin_port = htons((uint16_t)options->port)};
    if (inet_pton(AF_INET, options->host, &address.sin_addr) != 1 ||
        bind(listener, (struct sockaddr *)&address, sizeof(address)) || listen(listener, 8)) {
        perror("listen");
        close(listener);
        return -1;
    }
    pthread_t inference;
    if (pthread_create(&inference, NULL, worker, &s)) {
        close(listener);
        return -1;
    }
    fprintf(stderr, "HTTP listening on %s:%u; one resident model, one active transcription\n",
            options->host, options->port);
    while (!*stop) {
        struct pollfd p = {listener, POLLIN, 0};
        int rc = poll(&p, 1, 100);
        if (rc <= 0) {
            if (rc < 0 && errno != EINTR)
                break;
            continue;
        }
        int fd = accept(listener, NULL, NULL);
        if (fd < 0) {
            if (errno != EINTR)
                break;
            continue;
        }
        if (fcntl(fd, F_SETFL, O_NONBLOCK)) {
            close(fd);
            continue;
        }
#ifdef SO_NOSIGPIPE
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
        pthread_mutex_lock(&s.mutex);
        connection *c = NULL;
        for (int i = 0; i < CONNECTIONS; ++i)
            if (s.clients[i].fd < 0) {
                c = &s.clients[i];
                break;
            }
        if (c) {
            *c = (connection){.s = &s, .fd = fd};
            ++s.active;
            pthread_t thread;
            if (pthread_create(&thread, NULL, handle, c)) {
                c->fd = -1;
                --s.active;
                c = NULL;
            } else
                pthread_detach(thread);
        }
        pthread_mutex_unlock(&s.mutex);
        if (!c) {
            const char body[] =
                "{\"error\":{\"message\":\"Connection limit "
                "reached.\",\"type\":\"server_error\",\"param\":null,\"code\":\"server_busy\"}}";
            char response[512];
            int size =
                snprintf(response, sizeof(response),
                         "HTTP/1.1 503 Service Unavailable\r\n"
                         "Connection: close\r\nContent-Type: application/json\r\nRetry-After: 1\r\n"
                         "Content-Length: %zu\r\n\r\n%s",
                         sizeof(body) - 1, body);
#ifdef MSG_NOSIGNAL
            (void)send(fd, response, (size_t)size, MSG_NOSIGNAL);
#else
            (void)send(fd, response, (size_t)size, 0);
#endif
            close(fd);
        }
    }
    close(listener);
    pthread_mutex_lock(&s.mutex);
    for (int i = 0; i < CONNECTIONS; ++i)
        if (s.clients[i].fd >= 0)
            shutdown(s.clients[i].fd, SHUT_RDWR);
    while (s.active)
        pthread_cond_wait(&s.changed, &s.mutex);
    s.worker_stop = 1;
    pthread_cond_broadcast(&s.changed);
    pthread_mutex_unlock(&s.mutex);
    pthread_join(inference, NULL);
    pthread_cond_destroy(&s.changed);
    pthread_mutex_destroy(&s.mutex);
    return 0;
}
