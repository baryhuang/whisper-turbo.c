#!/bin/sh
# Entrypoint for Dockerfile.cloud: fetches the pinned upstream GGML
# checkpoint, verifies its SHA-256, converts it to the packed INT8
# WHTRBO format, and persists the result to a mounted volume — all on
# first container boot, so the image itself stays small and the model
# never has to be baked in or pre-mounted by the deploy step. Subsequent
# restarts with the same volume skip straight to serving.
set -eu

MODEL_DIR="/data"
MODEL_PATH="$MODEL_DIR/turbo-q8.whtrbo"
BIND_HOST="${BIND_HOST:-::}"
GGML_PATH="$MODEL_DIR/ggml-large-v3-turbo.bin"
GGML_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
GGML_SHA256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"

if [ ! -s "$MODEL_PATH" ]; then
  echo "[entrypoint] converted model missing at $MODEL_PATH; preparing it now" >&2
  mkdir -p "$MODEL_DIR"
  if [ ! -s "$GGML_PATH" ]; then
    echo "[entrypoint] downloading pinned GGML checkpoint..." >&2
    curl -fL -o "$GGML_PATH.part" "$GGML_URL"
    mv "$GGML_PATH.part" "$GGML_PATH"
  fi
  echo "$GGML_SHA256  $GGML_PATH" | sha256sum -c -
  echo "[entrypoint] converting to WHTRBO INT8..." >&2
  /usr/local/bin/import-ggml "$GGML_PATH" "$MODEL_PATH.part"
  mv "$MODEL_PATH.part" "$MODEL_PATH"
  rm -f "$GGML_PATH"
  echo "[entrypoint] model ready at $MODEL_PATH" >&2
fi

exec /usr/local/bin/whisper-turbo-server "$MODEL_PATH" "${PORT:-8080}" "$BIND_HOST"
