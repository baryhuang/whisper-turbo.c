CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror
PACKAGES = libcurl libcjson sndfile openssl
UNICODE_CFLAGS ?=
UNICODE_LIBS ?= -lunistring
.PHONY: all test
all: wer-bench
wer-bench: bench.c score.c score.h
	$(CC) $(CFLAGS) $(UNICODE_CFLAGS) $$(pkg-config --cflags $(PACKAGES)) bench.c score.c -o $@ $$(pkg-config --libs $(PACKAGES)) $(UNICODE_LIBS)
wer-test: ../../tests/wer_test.c score.c score.h
	$(CC) $(CFLAGS) $(UNICODE_CFLAGS) ../../tests/wer_test.c score.c -o $@ $(UNICODE_LIBS)
wer-cli-test: ../../tests/wer_cli_test.c
	$(CC) $(CFLAGS) $$(pkg-config --cflags libcjson) $< -o $@ $$(pkg-config --libs libcjson)
test: wer-bench wer-test wer-cli-test
	./wer-test
	./wer-cli-test ./wer-bench
