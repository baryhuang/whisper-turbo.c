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
