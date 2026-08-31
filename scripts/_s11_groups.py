"""CPU-only grouped cross-fitting and held-out k selection for S11.

No model, training loader, labels-based calibration, or legacy script is used.
Identity labels are used only to form disjoint groups and score retrieval.
"""

import numpy as np

try:
    from ._offline_common import evaluate, topk_mean, topk_values
except ImportError:  # Standalone scripts copied together into AutoDL scripts/.
    from _offline_common import evaluate, topk_mean, topk_values


SCAN_SEEDS = (2026, 2027, 2028)


def _compact(metrics):
    """Keep aggregate metrics, not per-query arrays, in the F/G JSON tables."""
    keys = ("R1", "R5", "R10", "mAP", "n_query", "skipped_no_positive", "gallery_size")
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key in keys
        if (value := metrics.get(key)) is not None
    }


def _counts(ann, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return {
        "identities": int(np.unique(ann.query_ids[indices]).size),
        "images": int(np.unique(ann.query_source_indices[indices]).size),
        "texts": int(indices.size),
        "queries": int(indices.size),
    }


def balanced_group_halves(groups, seed=2026):
    """GroupKFold-style two-way load balancing, without sklearn dependency.

    Whole groups are sorted by decreasing query count; seeded order breaks
    equal-size ties. Each next group goes to the fold with fewer queries.
    This balances sample counts while never splitting an image/identity.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) < 2:
        raise ValueError("group splitting requires a one-dimensional array of >=2 queries")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError("group splitting requires at least two distinct groups")
    randomized = np.random.default_rng(seed).permutation(len(unique))
    order = randomized[np.argsort(-counts[randomized], kind="stable")]
    assignments = np.empty(len(unique), dtype=np.int8)
    loads = [0, 0]
    for group_index in order:
        fold = 0 if loads[0] <= loads[1] else 1
        assignments[group_index] = fold
        loads[fold] += int(counts[group_index])
    halves = tuple(np.flatnonzero(assignments[inverse] == fold) for fold in (0, 1))
    _check_partition(halves[0], halves[1], len(groups))
    if np.intersect1d(groups[halves[0]], groups[halves[1]]).size:
        raise AssertionError("group splitter leaked a group across folds")
    return halves


def random_identity_halves(query_ids, seed):
    """Split shuffled identities in half, not shuffled query rows."""
    query_ids = np.asarray(query_ids)
    identities = np.unique(query_ids)
    if len(identities) < 2:
        raise ValueError("half-select-k requires at least two query identities")
    order = np.random.default_rng(seed).permutation(identities)
    midpoint = len(order) // 2
    first = np.flatnonzero(np.isin(query_ids, order[:midpoint]))
    second = np.flatnonzero(np.isin(query_ids, order[midpoint:]))
    _check_partition(first, second, len(query_ids))
    if np.intersect1d(query_ids[first], query_ids[second]).size:
        raise AssertionError("identity half selection leaked identities")
    return first, second


def _check_partition(first, second, n_query):
    both = np.concatenate((first, second))
    if not len(first) or not len(second):
        raise ValueError("both calibration and evaluation partitions must be nonempty")
    if len(both) != n_query or not np.array_equal(np.sort(both), np.arange(n_query)):
        raise AssertionError("calibration/evaluation must partition all query rows exactly once")


def sample_identity_groups(query_ids, pool, budget, seed):
    """Select complete identity groups until the query budget is met/exceeded.

    The returned actual number of queries can exceed the requested budget;
    reporting that difference is essential and no identity is truncated.
    The full endpoint returns the original pool order exactly.
    """
    pool = np.asarray(pool, dtype=np.int64)
    if not 0 < budget <= len(pool):
        raise ValueError(f"query calibration budget {budget} must be in [1, {len(pool)}]")
    if budget == len(pool):
        return pool.copy()
    pool_ids = np.asarray(query_ids)[pool]
    identities, counts = np.unique(pool_ids, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(identities))
    needed = int(np.searchsorted(np.cumsum(counts[order]), budget, side="left")) + 1
    selected_ids = identities[order[:needed]]
    selected = pool[np.isin(pool_ids, selected_ids)]
    if len(selected) < budget or not np.isin(selected, pool).all():
        raise AssertionError("identity-group sampling violated query budget or calibration pool")
    for identity in selected_ids:
        if np.count_nonzero(np.asarray(query_ids)[selected] == identity) != np.count_nonzero(pool_ids == identity):
            raise AssertionError("identity-group calibration sampling truncated an identity")
    return selected


def _overlaps(ann, calibration, evaluation):
    return {
        "queries": int(np.intersect1d(calibration, evaluation).size),
        "images": int(np.intersect1d(ann.query_source_indices[calibration], ann.query_source_indices[evaluation]).size),
        "identities": int(np.intersect1d(ann.query_ids[calibration], ann.query_ids[evaluation]).size),
    }


def _evaluate_setting(sim, ann, evaluation, penalty=None):
    return _compact(evaluate(sim, ann, queries=evaluation, penalty=penalty, alpha=0.5))


def _r1_only(sim, ann, queries, penalty, block_size=128):
    """Exact same top-1/tie policy without sorting for R1-only scans."""
    penalty = np.asarray(penalty, dtype=np.float32)
    if penalty.shape != (sim.shape[1],) or not np.isfinite(penalty).all():
        raise ValueError("R1 scan penalty must be a finite gallery-sized vector")
    correct = 0
    for start in range(0, len(queries), block_size):
        rows = queries[start:start + block_size]
        values = np.array(sim[rows], dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("R1 scan scores contain NaN/infinity")
        values -= np.float32(0.5) * penalty[None, :]
        predicted = values.argmax(axis=1)
        correct += int((ann.gallery_ids[predicted] == ann.query_ids[rows]).sum())
    return float(np.float32(correct) * np.float32(100) / np.float32(len(queries)))


def _crossfit_fold(sim, ann, calibration, evaluation, full_penalties, k, scheme, fold):
    if len(calibration) < k:
        raise ValueError(f"{scheme} fold {fold}: k={k} exceeds {len(calibration)} calibration queries")
    overlaps = _overlaps(ann, calibration, evaluation)
    if overlaps["queries"] or (scheme != "query_random" and overlaps["images"]):
        raise AssertionError(f"{scheme} fold {fold}: forbidden query/image overlap {overlaps}")
    if scheme == "identity_group" and overlaps["identities"]:
        raise AssertionError("identity crossfit calibration/evaluation identities overlap")
    penalties = topk_mean(sim, k, rows=calibration)
    result = {
        "fold": int(fold),
        "calibration": _counts(ann, calibration),
        "evaluation": _counts(ann, evaluation),
        "overlap": overlaps,
        "metrics": {
            "uncorrected": _evaluate_setting(sim, ann, evaluation),
            "crossfit_csls": _evaluate_setting(sim, ann, evaluation, penalties),
            "full_query_transductive_csls": _evaluate_setting(sim, ann, evaluation, full_penalties),
        },
    }
    values = result["metrics"]
    print(
        f"F {scheme:<15} fold={fold} cal={len(calibration):5d} eval={len(evaluation):5d} "
        f"raw={values['uncorrected']['R1']:.8f} crossfit={values['crossfit_csls']['R1']:.8f} "
        f"transductive={values['full_query_transductive_csls']['R1']:.8f} "
        f"cal_counts={result['calibration']} eval_counts={result['evaluation']} "
        f"overlap(images/ids)={overlaps['images']}/{overlaps['identities']}"
    )
    return result, penalties


def _scan_identities(sim, ann, calibration, evaluation, crossfit, crossfit_penalties, k, scan_sizes):
    rows = []
    sizes = []
    for size in scan_sizes:
        if isinstance(size, bool) or int(size) != size or size <= 0:
            raise ValueError(f"calibration query budget must be a positive integer: {size!r}")
        if int(size) not in sizes:
            sizes.append(int(size))
    for requested in [*sizes, "full"]:
        budget = len(calibration) if requested == "full" else requested
        if budget > len(calibration) or budget < k:
            rows.append({
                "query_budget": requested,
                "status": "skipped",
                "reason": f"requires k <= budget <= calibration size ({k} <= budget <= {len(calibration)})",
            })
            print(f"F scan budget={requested}: skipped (calibration pool={len(calibration)}, k={k})")
            continue
        repeats = []
        for seed in SCAN_SEEDS:
            subset = sample_identity_groups(ann.query_ids, calibration, budget, seed)
            overlaps = _overlaps(ann, subset, evaluation)
            if any(overlaps.values()):
                raise AssertionError("identity-group size scan leaked evaluation examples")
            if budget == len(calibration):
                if not np.array_equal(subset, calibration):
                    raise AssertionError("full scan endpoint changed calibration rows/order")
                penalties = crossfit_penalties
            else:
                penalties = topk_mean(sim, k, rows=subset)
            r1 = _r1_only(sim, ann, evaluation, penalties)
            exact = None
            if budget == len(calibration):
                exact = r1 == crossfit["metrics"]["crossfit_csls"]["R1"]
                if not exact:
                    raise AssertionError("full identity calibration scan differs from crossfit R1")
            repeats.append({
                "seed": seed,
                "actual_calibration": _counts(ann, subset),
                "budget_overshoot": int(len(subset) - budget),
                "R1": r1,
                "matches_crossfit_exactly": exact,
            })
        values = np.asarray([repeat["R1"] for repeat in repeats], dtype=np.float64)
        row = {
            "query_budget": requested,
            "status": "ok",
            "R1_mean": float(values.mean()),
            "R1_std": float(values.std(ddof=0)),
            "repetitions": repeats,
        }
        rows.append(row)
        actual = "/".join(str(repeat["actual_calibration"]["queries"]) for repeat in repeats)
        print(f"F scan budget={str(requested):>5} actual={actual:<18} R1={values.mean():.8f} +/- {values.std(ddof=0):.8f}")
    return rows


def _select_half(sim, ann, selection, evaluation, full_penalties, selection_ks, seed, direction):
    overlaps = _overlaps(ann, selection, evaluation)
    if any(overlaps.values()):
        raise AssertionError("held-out k selection leaked queries, images, or identities")
    if len(evaluation) < 10:
        raise ValueError("held-out half must contain at least 10 queries for the fixed-k10 reference")
    candidates = []
    best_k = None
    best_r1 = -np.inf
    best_penalty = None
    skipped_ks = []
    # Sorting makes the stated smallest-k tie break independent of CLI order.
    valid_ks = []
    for k in sorted(set(selection_ks)):
        if isinstance(k, bool) or int(k) != k or k <= 0:
            raise ValueError(f"selection k must be a positive integer: {k!r}")
        if k > min(len(selection), len(evaluation)):
            skipped_ks.append({"k": int(k), "reason": (
                f"exceeds selection/held-out bank query counts {len(selection)}/{len(evaluation)}; "
                "candidate eligibility uses only unlabeled counts")})
            continue
        valid_ks.append(int(k))
    if not valid_ks:
        raise ValueError("no k candidates fit the selection-half bank")
    # One column top-max(k) pass suffices for every candidate k. Each prefix
    # mean has the exact same sorted float32 reduction as topk_mean(k).
    top_values = topk_values(sim, max(valid_ks), rows=selection)
    for k in valid_ks:
        penalty = top_values[:k].mean(axis=0, dtype=np.float32)
        r1 = _r1_only(sim, ann, selection, penalty)
        candidates.append({"k": int(k), "selection_R1": r1})
        if r1 > best_r1:
            best_k, best_r1, best_penalty = int(k), r1, penalty
    if best_k is None:
        raise ValueError("no k candidates fit the selection-half bank")
    # Primary half-select protocol: k is locked before observing any held-out
    # labels. Estimate its penalty on the UNLABELED held-out query bank, as
    # usual for transductive CSLS; do not evaluate alternative k's here.
    # A single top-max(selected_k, 10) pass also serves the matched-bank k10
    # control, separating k-selection effects from calibration-bank effects.
    held_out_top = topk_values(sim, max(best_k, 10), rows=evaluation)
    held_out_penalty = held_out_top[:best_k].mean(axis=0, dtype=np.float32)
    matched_k10_penalty = held_out_top[:10].mean(axis=0, dtype=np.float32)
    held_out = _evaluate_setting(sim, ann, evaluation, held_out_penalty)
    matched_k10 = (dict(held_out) if best_k == 10 else
                   _evaluate_setting(sim, ann, evaluation, matched_k10_penalty))
    fixed_k10 = _evaluate_setting(sim, ann, evaluation, full_penalties)
    # Supplementary stricter variant: neither held-out texts nor held-out
    # labels enter calibration; freeze the selected H1 bank penalty on H2.
    frozen = _evaluate_setting(sim, ann, evaluation, best_penalty)
    result = {
        "seed": int(seed),
        "direction": direction,
        "selection": _counts(ann, selection),
        "evaluation": _counts(ann, evaluation),
        "overlap": overlaps,
        "candidates": candidates,
        "skipped_ks": skipped_ks,
        "selected_k": best_k,
        "selection_R1": best_r1,
        "locked_held_out": held_out,
        "locked_held_out_bank": "unlabeled held-out half; k locked using selection-half labels only",
        "fixed_held_out_bank_k10": matched_k10,
        "fixed_full_query_k10": fixed_k10,
        "delta_R1_vs_fixed_held_out_bank_k10": float(held_out["R1"] - matched_k10["R1"]),
        "delta_R1_vs_fixed_full_query_k10": float(held_out["R1"] - fixed_k10["R1"]),
        "supplementary_frozen_selection_bank": {
            "metrics": frozen,
            "delta_R1_vs_fixed_full_query_k10": float(frozen["R1"] - fixed_k10["R1"]),
            "note": "fixed selection-half r(g); difference includes calibration-bank distribution/size effects",
        },
    }
    print(
        f"G seed={seed} {direction} select={len(selection):5d} eval={len(evaluation):5d} "
        f"k={best_k:3d} heldout={held_out['R1']:.8f} "
        f"heldout-bank-k10={matched_k10['R1']:.8f} "
        f"delta-matched={result['delta_R1_vs_fixed_held_out_bank_k10']:+.8f} "
        f"full-k10={fixed_k10['R1']:.8f} delta-full={result['delta_R1_vs_fixed_full_query_k10']:+.8f} "
        f"frozen-selection-bank(supplement)={frozen['R1']:.8f}"
    )
    return result


def run_grouped(
    sim, ann, full_penalties, k=10, scan_sizes=(100, 500, 1000, 5000),
    selection_ks=(1, 5, 10, 20, 30, 50, 100), split_seeds=(2026, 2027, 2028, 2029, 2030),
):
    """Return JSON-safe task F/G results; keep all matrix work CPU/blocked."""
    if isinstance(k, bool) or int(k) != k or k <= 0:
        raise ValueError("crossfit k must be a positive integer")
    k = int(k)
    n_query, n_gallery = sim.shape
    if len(ann.query_ids) != n_query or len(ann.gallery_ids) != n_gallery:
        raise ValueError("similarity matrix dimensions do not match annotations")
    full_penalties = np.asarray(full_penalties)
    if full_penalties.shape != (n_gallery,) or not np.isfinite(full_penalties).all():
        raise ValueError("full-query CSLS penalties must be a finite gallery-sized vector")
    if n_query < 2:
        raise ValueError("cross-fitting requires at least two queries")
    if not selection_ks or not split_seeds:
        raise ValueError("half-select-k needs at least one candidate k and one split seed")

    query_order = np.random.default_rng(2026).permutation(n_query)
    query_halves = (query_order[: n_query // 2], query_order[n_query // 2 :])
    _check_partition(*query_halves, n_query)
    image_halves = balanced_group_halves(ann.query_source_indices, seed=2026)
    identity_halves = balanced_group_halves(ann.query_ids, seed=2026)
    schemes = {}
    identity_first = None
    identity_first_penalties = None
    for name, halves in (("query_random", query_halves), ("image_group", image_halves), ("identity_group", identity_halves)):
        folds = []
        for fold in (0, 1):
            calibration, evaluation = halves[fold], halves[1 - fold]
            result, penalties = _crossfit_fold(
                sim, ann, calibration, evaluation, full_penalties, k, name, fold
            )
            folds.append(result)
            if name == "identity_group" and fold == 0:
                identity_first, identity_first_penalties = result, penalties
        schemes[name] = {"folds": folds}
    scan = _scan_identities(
        sim, ann, identity_halves[0], identity_halves[1], identity_first,
        identity_first_penalties, k, scan_sizes,
    )

    # G's comparator is explicitly the fixed full-query k=10 baseline, even
    # when a caller uses another k for the task-F diagnostic.
    fixed_k10_penalties = full_penalties if k == 10 else topk_mean(sim, 10)
    selections = []
    for seed in split_seeds:
        first, second = random_identity_halves(ann.query_ids, seed)
        selections.append(_select_half(sim, ann, first, second, fixed_k10_penalties, selection_ks, seed, "H1_to_H2"))
        selections.append(_select_half(sim, ann, second, first, fixed_k10_penalties, selection_ks, seed, "H2_to_H1"))
    held_out_values = np.asarray([row["locked_held_out"]["R1"] for row in selections], dtype=np.float64)
    deltas = np.asarray([row["delta_R1_vs_fixed_full_query_k10"] for row in selections], dtype=np.float64)
    matched_deltas = np.asarray([row["delta_R1_vs_fixed_held_out_bank_k10"] for row in selections], dtype=np.float64)
    frozen_values = np.asarray([row["supplementary_frozen_selection_bank"]["metrics"]["R1"] for row in selections], dtype=np.float64)
    print(
        f"G SUMMARY primary_R1={held_out_values.mean():.8f} +/- {held_out_values.std(ddof=0):.8f}; "
        f"delta_matched_bank_k10={matched_deltas.mean():+.8f} +/- {matched_deltas.std(ddof=0):.8f}; "
        f"delta_full_query_k10={deltas.mean():+.8f} +/- {deltas.std(ddof=0):.8f}; "
        f"supplement_frozen_R1={frozen_values.mean():.8f} +/- {frozen_values.std(ddof=0):.8f}"
    )
    return {
        "F": {
            "k": k,
            "split_seed": 2026,
            "scan_seeds": list(SCAN_SEEDS),
            "std_ddof": 0,
            "protocol": (
                "Two reciprocal folds per scheme; all gallery images retained, no source exclusion. "
                "Legacy query split uses default_rng(2026).permutation and floor(n/2). "
                "Image/identity splits use whole-group GroupKFold-style decreasing-count greedy "
                "balancing with seeded tie order (not sklearn-version-dependent). Crossfit r(g) "
                "uses only calibration rows. CSLS ranking uses s - 0.5*r, equivalent to 2*s-r."
            ),
            "schemes": schemes,
            "calibration_scan_protocol": (
                "Evaluation is fixed to identity-group fold 1; calibration pool is fold 0. "
                "Budgets denote target query counts, not numbers of identities. Select complete "
                "identities in seeded random order until meeting/exceeding the budget. Report "
                "actual query counts/overshoot; never truncate an identity or sample evaluation rows. "
                "Full uses the exact calibration pool/order and must reproduce fold-0 crossfit R1."
            ),
            "calibration_scan": scan,
        },
        "G": {
            "split_seeds": [int(seed) for seed in split_seeds],
            "candidate_ks": [int(candidate) for candidate in sorted(set(selection_ks))],
            "std_ddof": 0,
            "protocol": (
                "Shuffle unique identities with the recorded seed; split identities into two halves. "
                "Choose k by selection-half R1, estimating each candidate r(g) from that same half. "
                "Tie-break to the smallest k. Lock k, estimate its r(g) using UNLABELED held-out "
                "queries, then evaluate that locked candidate once. Held-out labels never enter "
                "selection and alternative k candidates are not scored on the held-out half. "
                "Compare to fixed k=10 with the SAME held-out bank, isolating the k choice, and "
                "also to the requested fixed full-query k=10 reference. The latter difference "
                "includes calibration-bank effects; it is not a best-k search over full-query labels. "
                "Swap halves independently. Percentages throughout."
            ),
            "supplementary_protocol": (
                "Additionally freeze the selected selection-half r(g) for held-out evaluation. "
                "This stricter inductive variant uses neither held-out texts nor held-out labels "
                "for calibration, but is not the primary half-select-k result. Its difference "
                "from the full-query k=10 reference mixes k selection and bank effects."
            ),
            "repetitions": selections,
            "held_out_R1_mean": float(held_out_values.mean()),
            "held_out_R1_std": float(held_out_values.std(ddof=0)),
            "delta_R1_mean": float(deltas.mean()),
            "delta_R1_std": float(deltas.std(ddof=0)),
            "delta_R1_vs_matched_bank_k10_mean": float(matched_deltas.mean()),
            "delta_R1_vs_matched_bank_k10_std": float(matched_deltas.std(ddof=0)),
            "supplementary_frozen_selection_bank_R1_mean": float(frozen_values.mean()),
            "supplementary_frozen_selection_bank_R1_std": float(frozen_values.std(ddof=0)),
        },
    }
