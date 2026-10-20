#!/bin/bash
# One CHTC CPU job: linear probe for one (window file, model, feature set).
#
#   run_probe.sh <data_stem> <model> <pool> <shortcuts: yes|no> <n_repeats>
#
# Output: results_<data_stem>_<run>.tar.gz holding results/<data_stem>/<run>/oof_predictions.csv
# with <run> = probe_<model>_<pool>[_sc].
set -euo pipefail

DATA_STEM="$1"; MODEL="$2"; POOL="$3"; SHORTCUTS="$4"; N_REPEATS="$5"
WINDOW="${DATA_STEM%_ph*}"
RUN="probe_${MODEL}_${POOL}"; EXTRA=()
if [ "$SHORTCUTS" = "yes" ]; then RUN="${RUN}_sc"; EXTRA=(--add_shortcuts); fi
OUT="results/$DATA_STEM/$RUN"
TARBALL="results_${DATA_STEM}_${RUN}.tar.gz"
mkdir -p "$OUT"

finish() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "FAILED with exit code $status at $(date)" | tee "$OUT/FAILED" >&2
    fi
    tar czf "$TARBALL" "$OUT" || true
    exit "$status"
}
trap finish EXIT

export PYTHONPATH="$PWD/pydeps${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import sklearn, numpy" 2>/dev/null || \
    python3 -m pip install --quiet --no-cache-dir --target "$PWD/pydeps" numpy "scikit-learn>=1.8"

python3 scripts/probe.py --data "$DATA_STEM.npz" \
    --embeddings "${WINDOW}__${MODEL}.npz" \
    --pool "$POOL" --channels fhr_uc --n_repeats "$N_REPEATS" \
    --out_dir "$OUT" "${EXTRA[@]}"
test -s "$OUT/oof_predictions.csv"
