CC ?= cc
CPPFLAGS ?=
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -lm
# Optional for a compiler with OpenMP installed, e.g. OPENMP=-fopenmp.
OPENMP ?=

CORE := src/generic/whisper_turbo_image.c src/generic/whisper_turbo_encoder.c \
        src/generic/whisper_turbo_frontend.c src/generic/whisper_turbo_decoder.c
HEADERS := $(wildcard src/generic/*.h)
X86_CORE := src/generic/whisper_turbo_image.c src/generic/whisper_turbo_frontend.c \
            src/x86/whisper_turbo_encoder.c src/x86/whisper_turbo_decoder.c src/x86/whisper_turbo_q8.c
X86_HEADERS := $(wildcard src/x86/*.h)

.PHONY: all clean
all: build/whisper-turbo-transcribe build/whisper-turbo-encoder-bench

build:
	mkdir -p $@

build/whisper-turbo-transcribe: src/whisper_turbo_transcribe.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/whisper-turbo-encoder-bench: benchmarks/whisper_turbo_encoder_bench.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

clean:
	$(RM) build/whisper-turbo-transcribe build/whisper-turbo-encoder-bench

.PHONY: x86-tools
x86-tools: build/whisper-turbo-x86 build/import-ggml build/q8-bench build/w8a8-bench build/bench-health

build/whisper-turbo-x86: src/whisper_turbo_transcribe.c $(X86_CORE) $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -DWHISPER_X86 -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/import-ggml: tools/import_ggml.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/q8-bench: benchmarks/q8_bench.c benchmarks/encoder_reference.c src/x86/whisper_turbo_q8.c $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/bench-health: benchmarks/cloud/health.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

build/w8a8-bench: benchmarks/w8a8_bench.c benchmarks/encoder_reference.c src/x86/whisper_turbo_q8.c src/x86/whisper_turbo_w8a8.c $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)
