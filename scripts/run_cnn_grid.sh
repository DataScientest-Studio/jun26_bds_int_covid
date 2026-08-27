#!/usr/bin/env bash
#
# Run the full architecture x region grid unattended.
#
#   ./scripts/run_cnn_grid.sh
#   ./scripts/run_cnn_grid.sh --redundant-csv data/metadata/redundant_images.csv
#
# Any extra arguments are passed through to every run, so whatever you add
# applies uniformly across the grid -- which is the point of a grid.
#
# Safe to interrupt and restart: a cell whose metrics AND history files both
# already exist is skipped. Pass --force to re-run everything.

set -uo pipefail   # deliberately NOT -e: one failed cell must not kill the rest

ARCHITECTURES=(lenet scratch)
REGIONS=(full lungs background)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

REPORTS_DIR="reports/cnn"
LOG_DIR="logs/cnn_grid"
mkdir -p "$LOG_DIR"

FORCE=0
PASSTHROUGH=()
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE=1
    else
        PASSTHROUGH+=("$arg")
    fi
done

# Artifact names must match pipeline.DEFAULT_MODEL_NAMES and artifact_name().
model_name_for() {
    case "$1" in
        lenet)   echo "lenet" ;;
        scratch) echo "cnn_scratch" ;;
    esac
}

label_for() {
    local model_name region
    model_name="$(model_name_for "$1")"
    region="$2"
    if [[ "$region" == "full" ]]; then echo "$model_name"; else echo "${model_name}_${region}"; fi
}

started_at=$(date +%s)
echo "=========================================================="
echo "CNN grid started $(date '+%Y-%m-%d %H:%M:%S')"
echo "Architectures: ${ARCHITECTURES[*]}"
echo "Regions:       ${REGIONS[*]}"
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
    echo "Extra args:    ${PASSTHROUGH[*]}"
fi
echo "Logs:          $LOG_DIR/"
echo "=========================================================="
echo

declare -a RESULTS=()

for arch in "${ARCHITECTURES[@]}"; do
    for region in "${REGIONS[@]}"; do
        label="$(label_for "$arch" "$region")"
        metrics="$REPORTS_DIR/${label}_metrics.json"
        history="$REPORTS_DIR/${label}_history.json"
        log="$LOG_DIR/${arch}_${region}.log"

        # Requiring the history file too means runs from before history was
        # persisted are correctly treated as incomplete and redone.
        if [[ $FORCE -eq 0 && -f "$metrics" && -f "$history" ]]; then
            echo "[skip] $arch / $region  (already complete)"
            RESULTS+=("skip|$arch|$region|-")
            continue
        fi

        echo "[run ] $arch / $region  -> $log"
        cell_started=$(date +%s)

        python -m covid_xray.cnn \
            --architecture "$arch" \
            --region "$region" \
            "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" \
            > "$log" 2>&1
        status=$?

        cell_elapsed=$(( $(date +%s) - cell_started ))
        if [[ $status -eq 0 ]]; then
            echo "       done in ${cell_elapsed}s"
            RESULTS+=("ok|$arch|$region|${cell_elapsed}s")
        else
            echo "       FAILED (exit $status) after ${cell_elapsed}s -- see $log"
            echo "       last lines:"
            tail -5 "$log" | sed 's/^/         /'
            RESULTS+=("FAIL|$arch|$region|exit $status")
        fi
        echo
    done
done

total_elapsed=$(( $(date +%s) - started_at ))
echo "=========================================================="
echo "Grid finished $(date '+%Y-%m-%d %H:%M:%S') in ${total_elapsed}s"
echo "----------------------------------------------------------"
printf '%-6s %-9s %-12s %s\n' "STATUS" "ARCH" "REGION" "TIME"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r status arch region detail <<< "$row"
    printf '%-6s %-9s %-12s %s\n' "$status" "$arch" "$region" "$detail"
done
echo "=========================================================="
echo

# Comparison table, printed only if anything succeeded.
if compgen -G "$REPORTS_DIR/*_metrics.json" > /dev/null; then
    python - <<'PYEOF'
import json, pathlib
rows = {}
for path in sorted(pathlib.Path("reports/cnn").glob("*_metrics.json")):
    label = path.name.removesuffix("_metrics.json")
    region = next((r for r in ("lungs", "background") if label.endswith(f"_{r}")), "full")
    model = label.removesuffix(f"_{region}") if region != "full" else label
    payload = json.loads(path.read_text())
    if "test" in payload:
        rows.setdefault(model, {})[region] = payload["test"]["report"]["macro avg"]["f1-score"]

    history = path.with_name(f"{label}_history.json")
    if history.exists():
        epochs = len(json.loads(history.read_text())["loss"])
        rows.setdefault(model, {})[f"{region}_epochs"] = epochs

print("test macro F1 by region")
print(f"{'model':<14}{'full':>9}{'lungs':>9}{'background':>12}{'epochs (f/l/b)':>18}")
for model in sorted(rows):
    cell = lambda r: f"{rows[model][r]:.4f}" if r in rows[model] else "   -   "
    ep = lambda r: str(rows[model].get(f"{r}_epochs", "-"))
    epochs = "/".join(ep(r) for r in ("full", "lungs", "background"))
    print(f"{model:<14}{cell('full'):>9}{cell('lungs'):>9}{cell('background'):>12}{epochs:>18}")
    if "full" in rows[model] and "background" in rows[model]:
        gap = rows[model]["full"] - rows[model]["background"]
        print(f"{'':<14}full - background gap: {gap:+.4f}")
PYEOF
fi
