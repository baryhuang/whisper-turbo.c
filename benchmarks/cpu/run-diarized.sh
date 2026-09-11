#!/bin/sh
# One discarded warm-up, then two measured full-recording API requests.
set -eu
[ "$#" -eq 5 ] || { echo 'usage: run-diarized.sh SERVER MODEL DIAR_MODELS WAV RESULT_DIR' >&2; exit 2; }
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE WHISPER_ACTIVATIONS=int8
export WHISPER_DIARIZATION_MODELS=$3
mkdir "$5"
"$1" "$2" 8091 127.0.0.1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
attempt=0
until curl --noproxy '*' -fsS http://127.0.0.1:8091/health >/dev/null; do
    kill -0 "$server_pid"
    attempt=$((attempt+1))
    [ "$attempt" -lt 30 ] || exit 1
    sleep 1
done
for trial in 0 1 2; do
    printf 'trial=%s (0 is warm-up) ' "$trial"
    curl --noproxy '*' -fsS --max-time 150 \
        -F "file=@$4" -F model=gpt-4o-transcribe-diarize \
        -F language=en -F response_format=diarized_json -F temperature=0 \
        -o "$5/$trial.json" -w 'http=%{http_code} seconds=%{time_total}\n' \
        http://127.0.0.1:8091/v1/audio/transcriptions
done
echo 'Service-accounted lifetime memory peak (not reset per trial):'
cat /sys/fs/cgroup/memory.peak
