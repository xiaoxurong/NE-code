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
set -euo pipefail

DATA_STEM="$1"; REPEAT="$2"; CONFIG="$3"; RUN_NAME="$4"; shift 4

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

# Install only what the container lacks, into the job's scratch directory.
export PYTHONPATH="$PWD/pydeps${PYTHONPATH:+:$PYTHONPATH}"
MISSING=$(python3 - <<'PY'
import importlib.util
need = {"sklearn": "scikit-learn", "pandas": "pandas", "yaml": "pyyaml", "torch": "torch"}
print(" ".join(pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None))
PY
)
if [ -n "$MISSING" ]; then
    echo "installing: $MISSING"
    python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" $MISSING
fi
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

OUT="results/$DATA_STEM/$RUN_NAME/r$REPEAT"
python3 run.py --config "$CONFIG" \
    --root_path ./ --data_path "$DATA_STEM.npz" \
    --only_repeat "$REPEAT" \
    --results_dir "$OUT" --checkpoints ./ckpt/ \
    --num_workers 2 "$@"

test -s "$OUT/oof_predictions.csv"
tar czf "results_${DATA_STEM}_${RUN_NAME}_r${REPEAT}.tar.gz" "$OUT"
echo "done: $(date)"
