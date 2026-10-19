#!/bin/bash
# One CHTC GPU job: frozen embeddings from one pretrained encoder for one window file.
#
#   run_extract.sh <data_stem> <model>
#
# Output: emb_<window>__<model>.tar.gz holding embeddings/<window>__<model>.npz, where
# <window> is the data stem without its _ph<threshold> suffix (the windows, and so the
# embeddings, are the same for every label threshold).
set -euo pipefail

DATA_STEM="$1"; MODEL="$2"; shift 2   # any further arguments go to extract_embeddings.py
WINDOW="${DATA_STEM%_ph*}"
OUT="embeddings/${WINDOW}__${MODEL}.npz"
TARBALL="emb_${WINDOW}__${MODEL}.tar.gz"
mkdir -p embeddings

finish() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "FAILED with exit code $status at $(date)" | tee "embeddings/${WINDOW}__${MODEL}.FAILED" >&2
    fi
    tar czf "$TARBALL" embeddings || true
    exit "$status"
}
trap finish EXIT

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

export HF_HOME="$PWD/hf_cache"
export PYTHONPATH="$PWD/pydeps${PYTHONPATH:+:$PYTHONPATH}"
# momentfm pins very old numpy/transformers versions; install it without its pins and add
# what it actually imports. Tested with transformers 5.x and huggingface_hub 2.x.
# mantis-tsfm would pull a second copy of PyTorch, so it is installed without its pins too.
python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" --no-deps momentfm mantis-tsfm
python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" \
    transformers huggingface_hub safetensors einops datasets tqdm scikit-learn pandas
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import momentfm, mantis.trainer, transformers; print('momentfm, mantis, transformers', transformers.__version__, 'import fine')"

python3 scripts/extract_embeddings.py --data "$DATA_STEM.npz" --model "$MODEL" --out "$OUT" "$@"
test -s "$OUT"
echo "done: $(date)"
