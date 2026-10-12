#!/usr/bin/env bash
# Training-machine check: same formal schedule, two different candidate sets.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/revision/short_train_check.sh CONFIG SEED ITERS
Run block sizes 0 and 80 with one torchrun process and --debug_max_iters ITERS.
CONFIG retains its formal epoch count: ITERS only stops after successful updates.
DEBUG ITER logs show all losses, total loss, gradient norm, every group LR and
soft-label alpha. Assert finite values and equal LR/alpha/niter/loss weights.
Losses and gradients are not required to match across the two candidate settings.
Environment: PYTHON=python; REVISION_OUT_ROOT=/tmp/tagpr_revision
Both experiment names start debug_; new evidence directories never overwrite
existing runs. GPU training is to be executed on the training machine.
HELP
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi
PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
CONFIG=$("$PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_file(), p; print(p)' "$1")
SEED=$2
ITERS=$3
"$PYTHON" -c 'import sys; int(sys.argv[1]); assert int(sys.argv[2]) > 0, "ITERS must be positive"' "$SEED" "$ITERS"
OUT_ROOT=${REVISION_OUT_ROOT:-${TMPDIR:-/tmp}/tagpr_revision}
mkdir -p -- "$OUT_ROOT"
RUN_ROOT=$(mktemp -d "$OUT_ROOT/debug_short_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd)
echo "Short-training evidence directory: $RUN_ROOT"
cd -- "$REPO"

for block in 0 80; do
    mkdir -- "$RUN_ROOT/block$block"
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m torch.distributed.run \
      --standalone --nproc_per_node=1 main.py --config "$CONFIG" --seed "$SEED" \
      --itc_block_size "$block" --debug_max_iters "$ITERS" \
      --output_dir "$RUN_ROOT/block$block" --exp_name "debug_itc_block$block" \
      2>&1 | tee "$RUN_ROOT/block$block.log"
done

"$PYTHON" "$SCRIPT_DIR/compare_eval_outputs.py" train \
    --old-log "$RUN_ROOT/block0.log" --new-log "$RUN_ROOT/block80.log" \
    --expected-updates "$ITERS" --out "$RUN_ROOT/comparison.json"

"$PYTHON" - "$RUN_ROOT" "$ITERS" <<'PY'
from pathlib import Path
import sys
root, count = Path(sys.argv[1]), int(sys.argv[2])
for block in (0, 80):
    results = list((root / f'block{block}').glob('*/final_results.txt'))
    assert len(results) == 1, f'block{block}: missing/ambiguous final archive'
    text = results[0].read_text(encoding='utf-8')
    assert 'run_status: debug_stopped' in text, f'block{block}: not a clean debug stop'
    assert f'debug_max_iters: {count}' in text, f'block{block}: wrong requested stop'
    assert not (results[0].parent / 'checkpoint_best.pth').exists(), 'unmarked formal best in debug run'
print('Both runs exited normally and archived run_status: debug_stopped')
PY
echo "Short-training gate complete; retained evidence: $RUN_ROOT"
