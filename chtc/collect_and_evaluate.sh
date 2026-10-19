#!/bin/bash
# After the CHTC jobs finish: unpack every results_*.tar.gz, run the shortcut baselines
# on the same folds (if not done yet), and evaluate every run per window file.
#
# Run from the repository root:   bash chtc/collect_and_evaluate.sh
# Needs numpy, pandas, scipy, scikit-learn (no GPU, no PyTorch).
set -euo pipefail

N_REPEATS="${N_REPEATS:-5}"
N_FOLDS="${N_FOLDS:-5}"
REFERENCE="${REFERENCE:-missingness_length}"

shopt -s nullglob
tarballs=(results_*.tar.gz)
for t in "${tarballs[@]}"; do
    tar xzf "$t" && mkdir -p chtc/returned && mv "$t" chtc/returned/
done
echo "unpacked ${#tarballs[@]} job outputs"
failed=(results/*/*/r*/FAILED)
if [ ${#failed[@]} -gt 0 ]; then
    echo "WARNING: ${#failed[@]} job(s) failed; their repeats are missing from the evaluation:"
    printf '  %s\n' "${failed[@]%/FAILED}"
fi

for stem_dir in results/*/; do
    stem=$(basename "$stem_dir")
    data="dataset/$stem.npz"
    if [ ! -f "$data" ]; then
        echo "skip $stem: $data not found"; continue
    fi
    if [ ! -f "results/$stem/baselines/$REFERENCE/oof_predictions.csv" ]; then
        python3 scripts/baselines.py --data "$data" --n_repeats "$N_REPEATS" --n_folds "$N_FOLDS" \
            --out_dir "results/$stem/baselines"
    fi

    runs=()
    for d in "results/$stem"/*/; do
        name=$(basename "$d")
        [ "$name" = "baselines" ] && continue
        runs+=("$name=$d")
    done
    for d in "results/$stem/baselines"/*/; do
        runs+=("$(basename "$d")=$d")
    done

    echo
    echo "=== $stem ==="
    python3 scripts/evaluate.py --data "$data" --runs "${runs[@]}" \
        --reference "$REFERENCE" --subset all vaginal \
        --out "results/$stem/table.csv"
done
