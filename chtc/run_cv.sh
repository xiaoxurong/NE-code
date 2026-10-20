#!/bin/bash
# One CHTC job = one repeat of 5-fold cross-validation for one model and one window file.
#
#   run_cv.sh <data_stem> <repeat> <config> <run_name> [extra run.py args...]
#
# Inputs transferred into the job's working directory:
#   the code (run.py, exp/, model/, layers/, data_provider/, utils/, configs/)
#   <data_stem>.npz (a window file from scripts/prepare_ctu_uhb.py)
# Output: results_<data_stem>_<run_name>_r<repeat>.tar.gz holding
#   results/<data_stem>/<run_name>/r<repeat>/{oof_predictions.csv,cv_results.json}
# If the job fails, the tarball is still written (with a FAILED file) so HTCondor can
# return it, and the job exits non-zero so the submit file puts it on hold with a
# readable reason.
set -euo pipefail

DATA_STEM="$1"; REPEAT="$2"; CONFIG="$3"; RUN_NAME="$4"; shift 4
OUT="results/$DATA_STEM/$RUN_NAME/r$REPEAT"
TARBALL="results_${DATA_STEM}_${RUN_NAME}_r${REPEAT}.tar.gz"
mkdir -p "$OUT"

finish() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "FAILED with exit code $status at $(date); see the job's .out and .err logs" | tee "$OUT/FAILED" >&2
    fi
    tar czf "$TARBALL" "$OUT" || true
    exit "$status"
}
trap finish EXIT

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

# Matplotlib needs a writable config directory and no display.
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig"

# Install only what the container lacks, into the job's scratch directory.
# This list must cover every third-party import in run.py and the packages it loads.
export PYTHONPATH="$PWD/pydeps${PYTHONPATH:+:$PYTHONPATH}"
MISSING=$(python3 - <<'PY'
import importlib.util
need = {"numpy": "numpy", "torch": "torch", "sklearn": "scikit-learn>=1.8", "pandas": "pandas",
        "yaml": "pyyaml", "matplotlib": "matplotlib"}
print(" ".join(pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None))
PY
)
if [ -n "$MISSING" ]; then
    echo "installing: $MISSING"
    python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" $MISSING
fi
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Fail fast if anything run.py imports is still missing, before any training starts.
python3 -c "import run" || { echo "run.py cannot be imported; a dependency is missing" >&2; exit 3; }

python3 run.py --config "$CONFIG" \
    --root_path ./ --data_path "$DATA_STEM.npz" \
    --only_repeat "$REPEAT" \
    --results_dir "$OUT" --checkpoints ./ckpt/ \
    --num_workers 2 "$@"

test -s "$OUT/oof_predictions.csv"
echo "done: $(date)"
