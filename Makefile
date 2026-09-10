CC ?= cc
.DEFAULT_GOAL := all
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
            src/x86/whisper_turbo_encoder.c src/x86/whisper_turbo_decoder.c src/x86/whisper_turbo_q8.c \
            src/x86/whisper_turbo_attention.c src/x86/whisper_turbo_w8a8.c
X86_HEADERS := $(wildcard src/x86/*.h)
# These C files are included by wrappers, not separate translation units.
X86_INCLUDED := src/generic/whisper_turbo_encoder.c src/generic/whisper_turbo_decoder.c

.PHONY: all clean check

HTTP_CORE := src/server/api.c src/server/http.c src/server/response.c src/server/reference.c
DIAR_CORE := src/diarization/checkpoint.c src/diarization/network.c src/diarization/cluster.c src/diarization/audio.c src/diarization/pipeline.c
DIAR_HEADERS := $(wildcard src/diarization/*.h)
SERVER_CORE := $(HTTP_CORE) src/server/inference.c src/server/diarized.c src/server/alignment.c $(DIAR_CORE)
SERVER_HEADERS := $(wildcard src/server/*.h)
.PHONY: server
server: build/whisper-turbo-server

.PHONY: diarized-cli
diarized-cli: build/whisper-turbo-diarize
build/whisper-turbo-diarize: src/server/cli.c $(SERVER_CORE) $(SERVER_HEADERS) $(DIAR_HEADERS) $(X86_CORE) $(X86_INCLUDED) $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -DWHISPER_X86 -Isrc/server -Isrc/generic -Isrc/diarization $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) -pthread -lm -lz

build/resident-pipeline-bench: src/server/cli.c $(SERVER_CORE) $(SERVER_HEADERS) $(DIAR_HEADERS) $(X86_CORE) $(X86_INCLUDED) $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -DWT_RESIDENT_BENCH -DWHISPER_X86 -Isrc/server -Isrc/generic -Isrc/diarization $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) -pthread -lm -lz

build/alignment-test: tests/alignment_test.c src/server/alignment.c src/server/api.c src/server/response.c $(SERVER_HEADERS) $(DIAR_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/server -Isrc/diarization $(filter %.c,$^) -o $@ $(LDFLAGS) -lm

build/pipeline-test: tests/pipeline_test.c src/server/diarized.c src/server/alignment.c src/server/api.c src/server/response.c src/server/reference.c src/diarization/network.c src/diarization/checkpoint.c $(SERVER_HEADERS) $(DIAR_HEADERS) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/server -Isrc/diarization -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) -lm -lz

build/http-test: tests/http_test.c $(HTTP_CORE) $(SERVER_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/server $(filter %.c,$^) -o $@ $(LDFLAGS) -pthread -lm

build/http-model-test: tests/http_model_test.c src/server/api.c src/server/api.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/server $(filter %.c,$^) -o $@ $(LDFLAGS)

.PHONY: check-http
check-http: build/http-test build/diarized-api-test build/alignment-test build/pipeline-test
	./build/http-test
	./build/diarized-api-test
	./build/alignment-test
	./build/pipeline-test

build/diarized-api-test: tests/diarized_api_test.c src/server/api.c src/server/response.c src/server/reference.c $(SERVER_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/server $(filter %.c,$^) -o $@ $(LDFLAGS) -lm

build/whisper-turbo-server: src/server/main.c $(SERVER_CORE) $(SERVER_HEADERS) $(DIAR_HEADERS) $(X86_CORE) $(X86_INCLUDED) $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -DWHISPER_X86 -Isrc/server -Isrc/generic -Isrc/diarization $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) -pthread -lm -lz

build/inspect-diarization: tools/inspect_diarization.c src/diarization/checkpoint.c src/diarization/checkpoint.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/diarization $(filter %.c,$^) -o $@ $(LDFLAGS) -lz

.PHONY: diarization
diarization: build/diarize-community build/inspect-diarization
build/diarize-community: src/diarization/main.c $(DIAR_CORE) $(DIAR_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/diarization $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) -lm -lz

build/diarization-test: tests/diarization_test.c $(DIAR_CORE) $(DIAR_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/diarization $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) -lm -lz
.PHONY: check-diarization
check-diarization: build/diarization-test
	./build/diarization-test

build/diarization-fixture: tools/diarization_fixture.c src/diarization/audio.c src/diarization/audio.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/diarization $(filter %.c,$^) -o $@ $(LDFLAGS) -lm

build/wav-slice: tools/wav_slice.c src/diarization/audio.c src/diarization/audio.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc/diarization $(filter %.c,$^) -o $@ $(LDFLAGS) -lm

all: build/whisper-turbo-server build/whisper-turbo-transcribe build/whisper-turbo-encoder-bench

build:
	mkdir -p $@

build/whisper-turbo-transcribe: src/whisper_turbo_transcribe.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/whisper-turbo-encoder-bench: benchmarks/whisper_turbo_encoder_bench.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

clean:
	$(RM) build/whisper-turbo-diarize build/resident-pipeline-bench build/alignment-test build/pipeline-test
	$(RM) build/whisper-turbo-server build/http-test build/http-model-test build/diarized-api-test
	$(RM) build/whisper-turbo-transcribe build/whisper-turbo-encoder-bench \
	    build/whisper-turbo-x86 build/import-ggml build/q8-bench build/w8a8-bench \
	    build/attention-bench build/bench-health build/measure build/observe build/repeat-wav
	$(RM) build/diarize-community build/inspect-diarization build/diarization-test build/diarization-fixture build/wav-slice

.PHONY: x86-tools
x86-tools: build/whisper-turbo-x86 build/import-ggml build/q8-bench build/w8a8-bench build/bench-health

build/whisper-turbo-x86: src/whisper_turbo_transcribe.c $(X86_CORE) $(X86_INCLUDED) $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -DWHISPER_X86 -Isrc/generic $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/import-ggml: tools/import_ggml.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter %.c,$^) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/q8-bench: benchmarks/q8_bench.c benchmarks/encoder_reference.c src/x86/whisper_turbo_q8.c src/x86/whisper_turbo_w8a8.c src/generic/whisper_turbo_encoder.c $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/bench-health: benchmarks/cloud/health.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Linux-only: explicit target, not part of the portable default build.
build/measure: benchmarks/cloud/measure.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

build/observe: benchmarks/cloud/observe.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

build/repeat-wav: benchmarks/cloud/repeat_wav.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

build/w8a8-bench: benchmarks/w8a8_bench.c benchmarks/encoder_reference.c src/x86/whisper_turbo_q8.c src/x86/whisper_turbo_w8a8.c src/generic/whisper_turbo_encoder.c $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

build/attention-bench: benchmarks/attention_bench.c benchmarks/encoder_reference.c src/x86/whisper_turbo_q8.c src/x86/whisper_turbo_w8a8.c src/x86/whisper_turbo_attention.c src/generic/whisper_turbo_encoder.c $(HEADERS) $(X86_HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP) -Isrc/generic $(filter-out $(X86_INCLUDED),$(filter %.c,$^)) -o $@ $(LDFLAGS) $(OPENMP) $(LDLIBS)

check: build/q8-bench build/w8a8-bench build/attention-bench build/diarization-test build/http-test build/diarized-api-test build/alignment-test build/pipeline-test
	./build/q8-bench 1 0
	./build/w8a8-bench 1 0
	./build/attention-bench 1 1
	./build/attention-bench 7 1
	./build/attention-bench 64 1
	./build/diarization-test
	./build/http-test
	./build/diarized-api-test
	./build/alignment-test
	./build/pipeline-test
