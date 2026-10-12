#!/usr/bin/env bash
# Training-machine gate. Keep default Gumbel and the config's startup seed.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/revision/ab_eval_gate.sh OLD_COMMIT NEW_COMMIT CHECKPOINT CONFIG BACKBONE
Run both committed versions in detached worktrees, with one torchrun process,
identical trusted checkpoint/config, default router and no --eval_seed.
Both runs export fp32 scores. Assertions check the actual SequentialSampler and
model.eval state before image encoding. Then compare raw metrics, result arrays
and score values exactly. A difference fails without retry or protocol changes.

Environment: PYTHON=python; REVISION_OUT_ROOT=/tmp/tagpr_revision
Config anno_dir and image_dir must be absolute so both worktrees use the same data.
Run artifacts and worktrees are retained in a fresh directory; nothing is deleted.
HELP
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ $# -ne 5 ]]; then usage >&2; exit 2; fi

PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
OLD_COMMIT=$(git -C "$REPO" rev-parse --verify "$1^{commit}")
NEW_COMMIT=$(git -C "$REPO" rev-parse --verify "$2^{commit}")
CHECKPOINT=$("$PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_file(), p; print(p)' "$3")
CONFIG=$("$PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_file(), p; print(p)' "$4")
BACKBONE=$5
case "$BACKBONE" in ViT-B/16|ViT-L/14) ;; *) echo 'BACKBONE must be ViT-B/16 or ViT-L/14' >&2; exit 2;; esac
OUT_ROOT=${REVISION_OUT_ROOT:-${TMPDIR:-/tmp}/tagpr_revision}
mkdir -p -- "$OUT_ROOT"
RUN_ROOT=$(mktemp -d "$OUT_ROOT/ab_gate_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd)
echo "A/B evidence directory: $RUN_ROOT"

git -C "$REPO" worktree add --detach "$RUN_ROOT/worktree_old" "$OLD_COMMIT"
git -C "$REPO" worktree add --detach "$RUN_ROOT/worktree_new" "$NEW_COMMIT"

# The injected wrapper adds assertions/records only; neither checkout is edited.
# It never seeds RNGs and does not alter model mode, dataset, forward or scores.
cat > "$RUN_ROOT/assert_eval.py" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

worktree = Path(os.environ['TAGPR_AB_WORKTREE']).resolve()
os.chdir(worktree)
sys.path.insert(0, str(worktree))
import main
import model.tbps_model_MoE as model_module
from torch.utils.data import SequentialSampler

checkpoint, config_path, backbone, output_root, exp_name = sys.argv[1:]
checks = {'image_eval_calls': 0, 'shuffle_false': False, 'sequential_sampler': False}
original_build = main.build_pedes_data
original_factory = model_module.clip_vitb

def guarded_build(config):
    for key in ('anno_dir', 'image_dir'):
        if not Path(getattr(config, key)).is_absolute():
            raise ValueError('A/B config requires an absolute ' + key)
    dataset_name = str(getattr(config, 'dataset_name', 'TAG-PEDES')).lower().replace('_', '-')
    if dataset_name not in ('tag-pedes', 'tagpedes'):
        raise ValueError('this gate is for the existing TAG-PEDES path')
    checks['misc_seed'] = int(config.misc.seed)
    checks['router_mode'] = getattr(config.experiment, 'eval_router_mode', 'gumbel')
    checks['eval_seed'] = getattr(config.experiment, 'eval_seed', None)
    if checks['router_mode'] != 'gumbel' or checks['eval_seed'] is not None:
        raise ValueError('A/B gate requires official default Gumbel and no eval seed')
    if getattr(config.experiment, 'csls', False) or getattr(config.experiment, 'rerank', False):
        raise ValueError('A/B raw-score gate requires a config with CSLS/rerank disabled')
    loaders = original_build(config)
    test = loaders['test_loader']
    assert isinstance(test.sampler, SequentialSampler), 'evaluation must use SequentialSampler'
    checks['shuffle_false'] = True  # DataLoader maps shuffle=False to SequentialSampler.
    checks['sequential_sampler'] = True
    checks['test_batch_size'] = int(test.batch_size)
    checks['num_workers'] = int(test.num_workers)
    annotations = Path(config.anno_dir)
    files = [annotations] if annotations.is_file() else sorted(annotations.glob('*.json'))
    if not files:
        raise ValueError('no annotation JSON files found for A/B provenance')
    checks['annotation_hashes'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in files}
    return loaders

def guarded_factory(*args, **kwargs):
    model = original_factory(*args, **kwargs)
    original_encode = model.encode_image
    def guarded_encode(*positional, **keywords):
        assert not model.training, 'model must be in eval mode before image encoding'
        assert keywords.get('training', False) is False, 'no training image path in A/B eval'
        checks['image_eval_calls'] += 1
        return original_encode(*positional, **keywords)
    model.encode_image = guarded_encode
    return model

main.build_pedes_data = guarded_build
model_module.clip_vitb = guarded_factory
arguments = ['--eval_only', '--resume', checkpoint, '--config', config_path,
             '--backbone', backbone, '--dump_sim', '--dump_sim_dtype', 'float32',
             '--output_dir', output_root, '--exp_name', exp_name]
sys.argv = ['main.py', *arguments]
status = main.main(arguments)
assert status in (None, 0), status
assert checks['image_eval_calls'] > 0, 'image eval-state guard was never exercised'
results = list(Path(output_root).glob('*/sim_meta.json'))
assert len(results) == 1, 'expected exactly one new evaluation output directory'
with (results[0].parent / 'ab_runtime_checks.json').open('x', encoding='utf-8') as handle:
    json.dump(checks, handle, indent=2, allow_nan=False)
    handle.write('\n')
print('A/B runtime assertions passed: SequentialSampler, shuffle=False, model.eval')
PY

for side in old new; do
    mkdir -- "$RUN_ROOT/output_$side"
    TAGPR_AB_WORKTREE="$RUN_ROOT/worktree_$side" \
      PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m torch.distributed.run \
      --standalone --nproc_per_node=1 "$RUN_ROOT/assert_eval.py" \
      "$CHECKPOINT" "$CONFIG" "$BACKBONE" "$RUN_ROOT/output_$side" "ab_$side" \
      2>&1 | tee "$RUN_ROOT/$side.log"
done

OLD_DIR=$("$PYTHON" -c 'from pathlib import Path; import sys; p=list(Path(sys.argv[1]).glob("*/sim_meta.json")); assert len(p)==1; print(p[0].parent)' "$RUN_ROOT/output_old")
NEW_DIR=$("$PYTHON" -c 'from pathlib import Path; import sys; p=list(Path(sys.argv[1]).glob("*/sim_meta.json")); assert len(p)==1; print(p[0].parent)' "$RUN_ROOT/output_new")
"$PYTHON" "$SCRIPT_DIR/compare_eval_outputs.py" eval \
    --old-dir "$OLD_DIR" --new-dir "$NEW_DIR" --out "$RUN_ROOT/comparison.json"
echo "A/B gate complete; retained evidence and detached worktrees: $RUN_ROOT"
