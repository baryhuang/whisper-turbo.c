#!/bin/sh
# Foreground, single-model comparison. Results stay in the corpus directory.
set -eu
if [ "$#" -ne 4 ]; then
    echo "usage: sh run-case.sh SERVER MODEL CORPUS LABEL" >&2
    exit 2
fi
server=$1
model=$2
corpus=$3
label=$4
bench=${WER_BENCH:-/candidate/benchmarks/wer/wer-bench}
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE WHISPER_ACTIVATIONS=int8
echo "Starting $label: one resident model, eight threads, sequential requests"
"$server" "$model" 8091 127.0.0.1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
attempt=0
until curl --noproxy '*' -fsS http://127.0.0.1:8091/health >/dev/null; do
    kill -0 "$server_pid"
    attempt=$((attempt + 1))
    [ "$attempt" -lt 30 ] || exit 1
    sleep 1
done
"$bench" run "$corpus" http://127.0.0.1:8091/v1/audio/transcriptions "$label" native
"$bench" score "$corpus" "$label"
echo "Service cgroup memory: lifetime peak, not an independently reset measurement"
for counter in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak; do
    if [ -r "$counter" ]; then
        printf '%s ' "$counter"
        cat "$counter"
    fi
done
