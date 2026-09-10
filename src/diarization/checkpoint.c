#define _POSIX_C_SOURCE 200809L
#include "checkpoint.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

/* Deliberately bounded: no external calls, imports, object constructors,
 * path extraction, compressed checkpoint members, or arbitrary allocation
 * sizes specified by a pickle. Unknown opcodes are rejected. */
#define NODES 30000
#define ITEMS 16000
typedef struct node node;
struct node {
    int type;
    int64_t i;
    char *s;
    node **v;
    size_t n;
};
enum { NIL, INTEGER, STRING, GLOBAL, SEQ, DICT, OBJECT, TENSOR, STORAGE, MARK };
typedef struct {
    node *nodes;
    size_t used, references;
    node *stack[ITEMS], *memo[NODES];
    size_t sp;
    int bad;
} parser;
static uint16_t u16(const unsigned char *p) {
    return (uint16_t)(p[0] | p[1] << 8);
}
static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static int span(size_t off, size_t len, size_t total) {
    return off <= total && len <= total - off;
}
static node *make(parser *p, int type) {
    if (p->used == NODES) {
        p->bad = 1;
        return NULL;
    }
    node *n = &p->nodes[p->used++];
    n->type = type;
    return n;
}
static void push(parser *p, node *n) {
    if (!n || p->sp == ITEMS)
        p->bad = 1;
    else
        p->stack[p->sp++] = n;
}
static node *pop(parser *p) {
    if (!p->sp) {
        p->bad = 1;
        return NULL;
    }
    return p->stack[--p->sp];
}
static int append(parser *p, node *n, node **v, size_t count) {
    if (!n || n->n > ITEMS || count > ITEMS - n->n || count > 1000000 - p->references) {
        p->bad = 1;
        return -1;
    }
    node **a = realloc(n->v, (n->n + count + 1) * sizeof(*a));
    if (!a) {
        p->bad = 1;
        return -1;
    }
    n->v = a;
    memcpy(a + n->n, v, count * sizeof(*a));
    n->n += count;
    p->references += count;
    return 0;
}
static node *tuple(parser *p, size_t count) {
    if (count > p->sp) {
        p->bad = 1;
        return NULL;
    }
    node *n = make(p, SEQ);
    if (append(p, n, p->stack + p->sp - count, count))
        return NULL;
    p->sp -= count;
    return n;
}
static size_t mark(parser *p) {
    size_t i = p->sp;
    while (i && p->stack[i - 1]->type != MARK)
        --i;
    if (!i)
        p->bad = 1;
    return i;
}
static node *string(parser *p, const unsigned char *b, size_t n, int type) {
    if (n > 65536) {
        p->bad = 1;
        return NULL;
    }
    node *r = make(p, type);
    if (!r)
        return NULL;
    r->s = malloc(n + 1);
    if (!r->s) {
        p->bad = 1;
        return NULL;
    }
    memcpy(r->s, b, n);
    r->s[n] = 0;
    return r;
}
static node *get(node *n, const char *key) {
    if (!n || n->type != DICT)
        return NULL;
    for (size_t i = 0; i + 1 < n->n; i += 2)
        if (n->v[i]->s && !strcmp(n->v[i]->s, key))
            return n->v[i + 1];
    return NULL;
}
static node *parse(parser *p, const unsigned char *b, size_t size) {
    size_t at = 0;
    node *root = NULL;
    while (at < size && !p->bad) {
        unsigned op = b[at++];
        size_t len = 0, i = 0;
        node *n = NULL, *a = NULL, *f = NULL;
        switch (op) {
        case 0x80:
            if (at == size || b[at++] != 2)
                p->bad = 1;
            break;
        case '}':
            push(p, make(p, DICT));
            break;
        case ']':
        case ')':
            push(p, make(p, SEQ));
            break;
        case '(':
            push(p, make(p, MARK));
            break;
        case 'N':
            push(p, make(p, NIL));
            break;
        case 0x88:
        case 0x89:
            n = make(p, INTEGER);
            if (n)
                n->i = op == 0x88;
            push(p, n);
            break;
        case 'K':
        case 'M':
        case 'J':
            len = op == 'K' ? 1 : op == 'M' ? 2 : 4;
            if (!span(at, len, size)) {
                p->bad = 1;
                break;
            }
            n = make(p, INTEGER);
            if (n)
                n->i = len == 1 ? b[at] : len == 2 ? u16(b + at) : (int32_t)u32(b + at);
            at += len;
            push(p, n);
            break;
        case 'G':
            if (!span(at, 8, size)) {
                p->bad = 1;
                break;
            }
            at += 8;
            push(p, make(p, NIL));
            break;
        case 'X':
            if (!span(at, 4, size)) {
                p->bad = 1;
                break;
            }
            len = u32(b + at);
            at += 4;
            if (!span(at, len, size)) {
                p->bad = 1;
                break;
            }
            push(p, string(p, b + at, len, STRING));
            at += len;
            break;
        case 'c':
            i = at;
            while (at < size && b[at] != '\n')
                ++at;
            if (at < size)
                ++at;
            while (at < size && b[at] != '\n')
                ++at;
            if (at == size) {
                p->bad = 1;
                break;
            }
            push(p, string(p, b + i, at - i, GLOBAL));
            ++at;
            break;
        case 'q':
        case 'r':
        case 'h':
        case 'j':
            len = (op == 'q' || op == 'h') ? 1 : 4;
            if (!span(at, len, size)) {
                p->bad = 1;
                break;
            }
            i = len == 1 ? b[at] : u32(b + at);
            at += len;
            if (i >= NODES) {
                p->bad = 1;
                break;
            }
            if (op == 'q' || op == 'r') {
                if (!p->sp)
                    p->bad = 1;
                else
                    p->memo[i] = p->stack[p->sp - 1];
            } else
                push(p, p->memo[i]);
            break;
        case 't':
            i = mark(p);
            if (p->bad)
                break;
            n = tuple(p, p->sp - i);
            pop(p);
            push(p, n);
            break;
        case 0x85:
        case 0x86:
        case 0x87:
            n = tuple(p, op - 0x84);
            push(p, n);
            break;
        case 'Q':
            a = pop(p);
            n = make(p, STORAGE);
            if (n && a)
                append(p, n, &a, 1);
            push(p, n);
            break;
        case 'R':
        case 0x81:
            a = pop(p);
            f = pop(p);
            if (!a || !f || a->type != SEQ) {
                p->bad = 1;
                break;
            }
            if (f->s && !strcmp(f->s, "collections\nOrderedDict"))
                n = make(p, DICT);
            else if (f->s && !strcmp(f->s, "torch._utils\n_rebuild_tensor_v2")) {
                n = make(p, TENSOR);
                append(p, n, a->v, a->n);
            } else {
                n = make(p, OBJECT);
                if (n) {
                    node *v[] = {f, a};
                    append(p, n, v, 2);
                }
            }
            push(p, n);
            break;
        case 's':
            a = pop(p);
            f = pop(p);
            if (!p->sp || !a || !f) {
                p->bad = 1;
                break;
            }
            {
                node *v[] = {f, a};
                append(p, p->stack[p->sp - 1], v, 2);
            }
            break;
        case 'u':
        case 'e':
            i = mark(p);
            if (p->bad || i < 2) {
                p->bad = 1;
                break;
            }
            n = p->stack[i - 2];
            if (op == 'u' && (p->sp - i) % 2) {
                p->bad = 1;
                break;
            }
            append(p, n, p->stack + i, p->sp - i);
            p->sp = i - 1;
            break;
        case 'a':
            a = pop(p);
            if (!p->sp || !a) {
                p->bad = 1;
                break;
            }
            append(p, p->stack[p->sp - 1], &a, 1);
            break;
        case 'b':
            pop(p);
            if (!p->sp)
                p->bad = 1;
            break; /* inert instance attributes */
        case '.':
            if (p->sp != 1 || at != size)
                p->bad = 1;
            else
                root = p->stack[0];
            at = size;
            break;
        default:
            fprintf(stderr, "unsupported checkpoint opcode 0x%02x at %zu\n", op, at - 1);
            p->bad = 1;
        }
    }
    return p->bad ? NULL : root;
}
/* ZIP central directory avoids trusting data-descriptor local lengths. */
static const unsigned char *member(const diar_checkpoint *c, const char *suffix, size_t *bytes) {
    const unsigned char *b = c->mapping;
    size_t size = c->bytes, eocd = size;
    if (size < 22)
        return NULL;
    for (size_t i = size - 22;; --i) {
        if (u32(b + i) == 0x06054b50) {
            eocd = i;
            break;
        }
        if (!i || size - i > 65557)
            break;
    }
    if (eocd == size || u16(b + eocd + 4) || u16(b + eocd + 6))
        return NULL;
    size_t pos = u32(b + eocd + 16), n = u16(b + eocd + 10);
    const unsigned char *found = NULL;
    for (size_t i = 0; i < n; ++i) {
        if (!span(pos, 46, size) || u32(b + pos) != 0x02014b50)
            return NULL;
        size_t nl = u16(b + pos + 28), extra = u16(b + pos + 30), comment = u16(b + pos + 32);
        if (!span(pos + 46, nl + extra + comment, size))
            return NULL;
        size_t sl = strlen(suffix);
        const unsigned char *name = b + pos + 46;
        if (nl >= sl && !memcmp(name + nl - sl, suffix, sl)) {
            if (found || u16(b + pos + 10) != 0 || (u16(b + pos + 8) & 1))
                return NULL;
            size_t len = u32(b + pos + 24), off = u32(b + pos + 42);
            if (u32(b + pos + 20) != len || !span(off, 30, size) || u32(b + off) != 0x04034b50)
                return NULL;
            off += 30 + (size_t)u16(b + off + 26) + (size_t)u16(b + off + 28);
            if (!span(off, len, size) || len > UINT32_MAX)
                return NULL;
            if ((uint32_t)crc32(0, b + off, (uInt)len) != u32(b + pos + 16))
                return NULL;
            found = b + off;
            *bytes = len;
        }
        pos += 46 + nl + extra + comment;
    }
    return found;
}
void diar_checkpoint_close(diar_checkpoint *c) {
    if (c->mapping)
        munmap(c->mapping, c->bytes);
    memset(c, 0, sizeof(*c));
}
int diar_checkpoint_open(diar_checkpoint *c, const char *path) {
    memset(c, 0, sizeof(*c));
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1)
        return -1;
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0)
        return -1;
    if (fstat(fd, &st) || st.st_size < 22 || st.st_size > 512 * 1024 * 1024) {
        close(fd);
        return -1;
    }
    c->bytes = (size_t)st.st_size;
    c->mapping = mmap(NULL, c->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (c->mapping == MAP_FAILED) {
        c->mapping = NULL;
        return -1;
    }
    size_t len = 0;
    const unsigned char *data = member(c, "/byteorder", &len);
    if (!data || len != 6 || memcmp(data, "little", 6))
        goto fail;
    data = member(c, "/data.pkl", &len);
    if (!data || len > 1024 * 1024)
        goto fail;
    parser *p = calloc(1, sizeof(*p));
    if (!p)
        goto fail;
    p->nodes = calloc(NODES, sizeof(node));
    if (!p->nodes) {
        free(p);
        goto fail;
    }
    node *root = parse(p, data, len), *dict = get(root, "state_dict");
    int result = -1;
    if (!dict || dict->type != DICT || dict->n % 2)
        goto parsed;
    for (size_t i = 0; i < dict->n; i += 2) {
        node *key = dict->v[i], *v = dict->v[i + 1];
        if (key->type != STRING || !key->s || strlen(key->s) >= sizeof(c->tensors[0].name) ||
            v->type != TENSOR || v->n != 6 || c->count == 512)
            goto parsed;
        node *s = v->v[0], *off = v->v[1], *shape = v->v[2], *stride = v->v[3];
        if (s->type != STORAGE || s->n != 1 || off->type != INTEGER || off->i < 0 ||
            shape->type != SEQ || stride->type != SEQ || shape->n != stride->n || shape->n > 4)
            goto parsed;
        s = s->v[0];
        if (s->type != SEQ || s->n != 5 || !s->v[0]->s || strcmp(s->v[0]->s, "storage") ||
            !s->v[1]->s || !s->v[2]->s || s->v[4]->type != INTEGER || s->v[4]->i < 0)
            goto parsed;
        int dtype = !strcmp(s->v[1]->s, "torch\nFloatStorage")  ? 4
                    : !strcmp(s->v[1]->s, "torch\nLongStorage") ? 8
                                                                : 0;
        if (!dtype)
            goto parsed;
        diar_tensor *t = &c->tensors[c->count];
        strcpy(t->name, key->s);
        t->rank = (int)shape->n;
        t->dtype = dtype;
        t->count = 1;
        for (size_t d = shape->n; d-- > 0;) {
            if (shape->v[d]->type != INTEGER || shape->v[d]->i <= 0 ||
                stride->v[d]->type != INTEGER || stride->v[d]->i != (int64_t)t->count ||
                shape->v[d]->i > 10000000 || t->count > 100000000 / (size_t)shape->v[d]->i)
                goto parsed;
            t->shape[d] = (size_t)shape->v[d]->i;
            t->count *= t->shape[d];
        }
        char suffix[128];
        if (snprintf(suffix, sizeof(suffix), "/data/%s", s->v[2]->s) >= (int)sizeof(suffix))
            goto parsed;
        data = member(c, suffix, &len);
        if (!data || (uintptr_t)data % 4 || (uint64_t)s->v[4]->i != len / (unsigned)dtype ||
            len % (unsigned)dtype || !span((size_t)off->i, t->count, len / (unsigned)dtype))
            goto parsed;
        for (size_t k = 0; k < c->count; ++k)
            if (!strcmp(c->tensors[k].name, t->name))
                goto parsed;
        t->data = data + (size_t)off->i * (unsigned)dtype;
        ++c->count;
    }
    result = 0;
parsed:
    for (size_t i = 0; i < p->used; ++i) {
        free(p->nodes[i].s);
        free(p->nodes[i].v);
    }
    free(p->nodes);
    free(p);
    if (!result)
        return 0;
fail:
    fprintf(stderr, "invalid or unsupported checkpoint: %s\n", path);
    diar_checkpoint_close(c);
    return -1;
}
const diar_tensor *diar_tensor_find(const diar_checkpoint *c, const char *name) {
    for (size_t i = 0; i < c->count; ++i)
        if (!strcmp(c->tensors[i].name, name))
            return &c->tensors[i];
    return NULL;
}
const float *diar_weights(const diar_checkpoint *c, const char *name, int rank,
                          const size_t *shape) {
    const diar_tensor *t = diar_tensor_find(c, name);
    if (!t || t->dtype != 4 || t->rank != rank)
        return NULL;
    for (int i = 0; i < rank; ++i)
        if (t->shape[i] != shape[i])
            return NULL;
    return t->data;
}
int diar_npz_read(const char *path, const char *name, int rank, const size_t *shape, double *out) {
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1 || rank < 1 || rank > 2)
        return -1;
    diar_checkpoint c = {0};
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0)
        return -1;
    if (fstat(fd, &st) || st.st_size < 22 || st.st_size > 1024 * 1024) {
        close(fd);
        return -1;
    }
    c.bytes = (size_t)st.st_size;
    c.mapping = mmap(NULL, c.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (c.mapping == MAP_FAILED)
        return -1;
    size_t len = 0;
    const unsigned char *p = member(&c, name, &len);
    int result = -1;
    if (!p || len < 10 || memcmp(p, "\223NUMPY\1\0", 8))
        goto done;
    size_t header = u16(p + 8);
    if (header > 512 || !span(10, header, len))
        goto done;
    char h[513];
    memcpy(h, p + 10, header);
    h[header] = 0;
    int dtype = strstr(h, "'<f4'") ? 4 : strstr(h, "'<f8'") ? 8 : 0;
    if (!dtype || !strstr(h, "'fortran_order': False"))
        goto done;
    char *s = strstr(h, "'shape': (");
    if (!s)
        goto done;
    s += 10;
    size_t count = 1;
    for (int i = 0; i < rank; ++i) {
        char *end;
        unsigned long value = strtoul(s, &end, 10);
        if (end == s || value != shape[i] || value == 0 || count > 1000000 / value)
            goto done;
        count *= value;
        s = end;
        while (*s == ' ' || *s == ',')
            ++s;
    }
    if (*s != ')' || len - 10 - header != count * (unsigned)dtype)
        goto done;
    for (size_t i = 0; i < count; ++i) {
        if (dtype == 4) {
            float f;
            memcpy(&f, p + 10 + header + i * 4, 4);
            out[i] = f;
        } else
            memcpy(out + i, p + 10 + header + i * 8, 8);
    }
    result = 0;
done:
    diar_checkpoint_close(&c);
    return result;
}
