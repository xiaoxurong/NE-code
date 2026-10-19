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
# Version pins, tested together:
#  - transformers 5.x silently disables itself with PyTorch < 2.6 ("T5EncoderModel requires
#    the PyTorch library"), and the container has PyTorch 2.4.1, so transformers stays on 4.46.
#  - transformers 4.46 needs huggingface_hub < 1.0; recent `datasets` (imported by Mantis)
#    needs a newer hub, so datasets is pinned to 4.0.0, which works with hub 0.26.
#  - momentfm and mantis-tsfm are installed without their own pins: momentfm pins
#    numpy 1.25 / transformers 4.33, and mantis-tsfm would pull a second copy of PyTorch.
python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" --no-deps momentfm mantis-tsfm
python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" \
    "transformers==4.46.3" "huggingface_hub==0.26.5" "datasets==4.0.0" \
    safetensors einops tqdm scikit-learn pandas
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 - <<'PY'
import transformers, huggingface_hub, momentfm, mantis.trainer
from transformers.utils import is_torch_available
print("transformers", transformers.__version__, "| hub", huggingface_hub.__version__)
assert is_torch_available(), "transformers cannot use this PyTorch; check the version pins"
PY

python3 scripts/extract_embeddings.py --data "$DATA_STEM.npz" --model "$MODEL" --out "$OUT" "$@"
test -s "$OUT"
echo "done: $(date)"
