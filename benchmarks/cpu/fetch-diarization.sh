#!/bin/sh
# Download the separately licensed checkpoints at runtime, never into an image.
set -eu
: "${HF_TOKEN:?HF_TOKEN must have accepted Community-1 access}"
destination=${1:?usage: sh fetch-diarization.sh DESTINATION}
mkdir -p "$destination"
revision=3533c8cf8e369892e6b79ff1bf80f7b0286a54ee
while read -r upstream digest; do
    filename=$(printf '%s' "$upstream" | tr / -)
    target="$destination/$filename"
    curl -fsSL --retry 2 --max-time 90 \
        -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/$revision/$upstream" \
        -o "$target"
    printf '%s  %s\n' "$digest" "$target" | sha256sum -c -
done <<'CHECKPOINTS'
segmentation/pytorch_model.bin 7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e
embedding/pytorch_model.bin 6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929
plda/xvec_transform.npz 325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f
plda/plda.npz 9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255
CHECKPOINTS
