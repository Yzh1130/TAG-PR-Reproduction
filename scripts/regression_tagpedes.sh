#!/usr/bin/env bash
# Numerical regression harness for the unchanged TAG-PEDES evaluation path.
# Run this script once before and once after a candidate change with the same
# checkpoint/config/environment, then compare all four values and the matrix
# digest printed at the end.

set -euo pipefail

: "${TAGPEDES_CHECKPOINT:?set TAGPEDES_CHECKPOINT to an existing TAG-PEDES checkpoint}"

TAGPEDES_CONFIG="${TAGPEDES_CONFIG:-config/config.yaml}"
TAGPEDES_OUTPUT_ROOT="${TAGPEDES_OUTPUT_ROOT:-regression_tagpedes}"
TAGPEDES_NPROC="${TAGPEDES_NPROC:-1}"
TAGPEDES_PYTHON="${TAGPEDES_PYTHON:-python}"

if [[ ! -f "${TAGPEDES_CHECKPOINT}" ]]; then
  echo "ERROR: TAGPEDES_CHECKPOINT does not exist: ${TAGPEDES_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${TAGPEDES_CONFIG}" ]]; then
  echo "ERROR: TAGPEDES_CONFIG does not exist: ${TAGPEDES_CONFIG}" >&2
  exit 2
fi
if ! [[ "${TAGPEDES_NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TAGPEDES_NPROC must be a positive integer" >&2
  exit 2
fi

run_stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
run_dir="${TAGPEDES_OUTPUT_ROOT%/}/${run_stamp}"
if [[ -e "${run_dir}" ]]; then
  echo "ERROR: refusing to reuse regression output directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

echo "TAG-PEDES numerical regression"
echo "checkpoint=${TAGPEDES_CHECKPOINT}"
echo "config=${TAGPEDES_CONFIG}"
echo "output_dir=${run_dir}"

"${TAGPEDES_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${TAGPEDES_NPROC}" \
  main.py \
  --config "${TAGPEDES_CONFIG}" \
  --eval_only \
  --resume "${TAGPEDES_CHECKPOINT}" \
  --output_dir "${run_dir}" \
  --dump_sim \
  --dump_sim_dtype float32 \
  --no_csls \
  --no_rerank \
  2>&1 | tee "${run_dir}/regression.log"

"${TAGPEDES_PYTHON}" - "${run_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
meta_path = run_dir / "sim_meta.json"
sim_path = run_dir / "sim_qg.npy"
if not meta_path.is_file() or not sim_path.is_file():
    raise SystemExit(
        f"ERROR: expected {meta_path} and {sim_path} after evaluation"
    )
meta = json.loads(meta_path.read_text(encoding="utf-8"))
metrics = meta["raw_metrics"]
digest = hashlib.sha256()
with sim_path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)

print("TAG-PEDES REGRESSION RESULT")
for key in ("R@1", "R@5", "R@10", "mAP"):
    print(f"{key}={metrics[key]!r}")
print(f"sim_qg_file_sha256_16={digest.hexdigest()[:16]}")
print(f"sim_qg_path={sim_path.resolve()}")
print("END TAG-PEDES REGRESSION RESULT")
PY
