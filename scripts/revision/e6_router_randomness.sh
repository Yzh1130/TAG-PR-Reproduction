#!/usr/bin/env bash
# GPU eval-only repetitions. No training, tuning, rerouting or metric thresholds.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/revision/e6_router_randomness.sh CONFIG BACKBONE CHECKPOINT OUT_ROOT N_GUMBEL_SEEDS
Run argmax once (eval_seed=0), then Gumbel with eval_seed=0..N-1.
Every run has its own exclusive directory/exp_name; nothing is overwritten.
Report exact raw R1/R5/R10/mAP, differences from argmax and the observed
maximum pairwise difference among THESE N repetitions (not a population bound).
Environment: PYTHON=python. Requires the real TAG-PEDES GPU environment.
Only --help and shell syntax tests run locally; no training is performed.
HELP
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ $# -ne 5 ]]; then usage >&2; exit 2; fi
PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
CONFIG=$("$PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_file(), p; print(p)' "$1")
BACKBONE=$2
CHECKPOINT=$("$PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_file(), p; print(p)' "$3")
OUT_ROOT=$4
N=$5
case "$BACKBONE" in ViT-B/16|ViT-L/14) ;; *) echo 'BACKBONE must be ViT-B/16 or ViT-L/14' >&2; exit 2;; esac
"$PYTHON" -c 'import sys; assert int(sys.argv[1]) > 0, "N_GUMBEL_SEEDS must be positive"' "$N"
case "/$OUT_ROOT/" in */_archive_evidence/*|*/_dump/*) echo 'Choose a NEW output root outside immutable archives' >&2; exit 2;; esac
mkdir -p -- "$OUT_ROOT"
RUN_ROOT=$(mktemp -d "$OUT_ROOT/e6_router_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd)
echo "Router evidence directory: $RUN_ROOT"
cd -- "$REPO"

run_eval() {
    local router=$1 seed=$2 label=$3
    mkdir -- "$RUN_ROOT/$label"
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m torch.distributed.run \
      --standalone --nproc_per_node=1 main.py --eval_only --config "$CONFIG" \
      --backbone "$BACKBONE" --resume "$CHECKPOINT" \
      --eval_router_mode "$router" --eval_seed "$seed" --no_csls --no_rerank \
      --output_dir "$RUN_ROOT/$label" --exp_name "e6_$label" \
      2>&1 | tee "$RUN_ROOT/$label.log"
}
run_eval argmax 0 argmax
for ((seed=0; seed<N; seed++)); do run_eval gumbel "$seed" "gumbel_seed$seed"; done

# This exact aggregator is also exercised on derived 200x100 MOCK metrics.
"$PYTHON" - "$RUN_ROOT" "$N" "$CONFIG" "$CHECKPOINT" "$BACKBONE" "$REPO" "$OUT_ROOT" <<'PY_SUMMARY'
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import yaml

root = Path(sys.argv[1])
count = int(sys.argv[2])
fields = ("r1", "r5", "r10", "mAP")
inputs = {}

def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    inputs[str(path)] = {"bytes": path.stat().st_size, "sha256_16": digest.hexdigest()[:16]}

def read_run(label, mode, seed):
    matches = list((root / label).glob("*/evaluation_revision.json"))
    if len(matches) != 1:
        raise ValueError(f"{label}: expected ONE new evaluation_revision.json, got {matches}")
    record(matches[0])
    config_used = matches[0].parent / "config_used.yaml"
    record(config_used)
    effective_config = yaml.safe_load(config_used.read_text(encoding="utf-8"))
    startup_seed = int(effective_config["misc"]["seed"])
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    if payload["eval_router_mode"] != mode or payload["eval_seed"] != seed:
        raise ValueError(f"{label}: router/seed provenance mismatch")
    if payload.get("csls") is not None or payload.get("reranked") is not None:
        raise ValueError(f"{label}: expected raw metrics without CSLS/reranking")
    metrics = {key: float(payload["original"][key]) for key in fields}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError(f"{label}: nonfinite metric")
    return {"label": label, "eval_router_mode": mode, "eval_seed": seed,
            "startup_misc_seed": startup_seed,
            "metrics": metrics, "evaluation_json": str(matches[0]),
            "config_used": str(config_used)}

record(sys.argv[3])
record(sys.argv[4])
baseline = read_run("argmax", "argmax", 0)
gumbel = [read_run(f"gumbel_seed{s}", "gumbel", s) for s in range(count)]
for run in gumbel:
    run["minus_argmax_pp"] = {key: run["metrics"][key] - baseline["metrics"][key] for key in fields}
maximum = {key: max(r["metrics"][key] for r in gumbel) - min(r["metrics"][key] for r in gumbel) for key in fields}
summary = {"status": "complete", "task": "e6_router_randomness", "argmax": baseline,
           "gumbel": gumbel, "n_repetitions": count,
           "observed_maximum_pairwise_difference_pp": maximum,
           "maximum_scope": f"n={count} 次重复中观测到的最大差异; not a bound over other seeds",
           "provenance": {"argv": ["bash", "scripts/revision/e6_router_randomness.sh", sys.argv[3], sys.argv[5], sys.argv[4], sys.argv[7], str(count)],
                          "seeds": {"argmax_eval": 0, "gumbel_eval": list(range(count)),
                                    "startup_misc_seed_by_run": {r["label"]: r["startup_misc_seed"] for r in [baseline, *gumbel]}},
                          "inputs": inputs, "git_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sys.argv[6], text=True).strip()},
           "training_executed": False, "postprocessing_enabled": False}
with (root / "router_randomness.json").open("x", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
with (root / "router_randomness.log").open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
PY_SUMMARY
echo "E6 complete; retained evidence: $RUN_ROOT"
