"""Offline source-domain error enrichment, with query-level bootstrap CIs.

The enrichment denominator is the mean conditional prior over the *same error
queries* used by the numerator, not the unconditional gallery composition.
No training or evaluation modules are imported by this tool.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._analysis_common import (
    DOMAIN_NAMES,
    annotation_from_records,
    load_annotations,
    make_mock_annotation,
    print_table,
    write_json,
)


BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 2026


def parse_predictions(records, ann):
    """Read misc.eval.metric_eval's [{query_index: {match, pred}}] format.

    Validate every stored match against annotation identities to catch stale or
    mismatched snapshots before computing an apparently plausible statistic.
    """
    n_query, n_gallery = len(ann.query_ids), len(ann.gallery_ids)
    if not isinstance(records, list) or len(records) != n_query:
        raise ValueError(f"prediction snapshot must contain exactly {n_query} query records")
    top1 = np.full(n_query, -1, dtype=np.int64)
    for position, item in enumerate(records):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"prediction record {position} must have one query-index key")
        key, value = next(iter(item.items()))
        try:
            query_index = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"invalid query index {key!r}") from None
        if str(query_index) != str(key) or not 0 <= query_index < n_query:
            raise ValueError(f"query index {key!r} is not in [0, {n_query})")
        if top1[query_index] != -1:
            raise ValueError(f"duplicate prediction for query {query_index}")
        if not isinstance(value, dict):
            raise ValueError(f"query {query_index}: prediction payload must be an object")
        predictions, matches = value.get("pred"), value.get("match")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"query {query_index}: pred must be a nonempty gallery-index list")
        if any(isinstance(index, bool) or not isinstance(index, int)
               or not 0 <= index < n_gallery for index in predictions):
            raise ValueError(f"query {query_index}: pred contains an invalid gallery index")
        if len(set(predictions)) != len(predictions):
            raise ValueError(f"query {query_index}: pred contains duplicate gallery indices")
        if (not isinstance(matches, list) or len(matches) != len(predictions)
                or any(match not in (0, 1) for match in matches)):
            raise ValueError(f"query {query_index}: match must contain one 0/1 per prediction")
        expected = (ann.gallery_ids[predictions] == ann.query_ids[query_index]).astype(int)
        if not np.array_equal(expected, matches):
            raise ValueError(
                f"query {query_index}: snapshot match flags disagree with annotation IDs; "
                "check the annotation/query/gallery ordering and snapshot experiment"
            )
        top1[query_index] = predictions[0]
    if np.any(top1 < 0):
        raise ValueError("prediction snapshot has missing query indices")
    return top1


def paired_bootstrap_enrichment(same_source, priors):
    """Resample (same-source-error indicator, conditional prior) query pairs.

    Blocks of 64 replicates bound index workspace to about 8 MiB at 16,504
    queries. Percentile intervals are undefined when a replicate has a zero
    denominator; such a case is reported rather than silently discarded.
    """
    same_source = np.asarray(same_source, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    if same_source.ndim != 1 or priors.shape != same_source.shape:
        raise ValueError("bootstrap indicators and priors must be same-length vectors")
    if not len(priors) or priors.mean() == 0:
        return None, 0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ratios = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    invalid = 0
    for start in range(0, BOOTSTRAP_REPEATS, 64):
        end = min(start + 64, BOOTSTRAP_REPEATS)
        indices = rng.integers(0, len(priors), size=(end - start, len(priors)))
        denominator = priors[indices].sum(axis=1)
        numerator = same_source[indices].sum(axis=1)
        valid = denominator > 0
        invalid += int((~valid).sum())
        ratios[start:end] = np.divide(
            numerator, denominator, out=np.zeros(end - start), where=valid
        )
    if invalid:
        return None, invalid
    return np.quantile(ratios, [0.025, 0.975]).tolist(), 0


def analyze_errors(top1, ann):
    top1 = np.asarray(top1)
    if (top1.shape != ann.query_ids.shape or not np.issubdtype(top1.dtype, np.integer)
            or np.any(top1 < 0) or np.any(top1 >= len(ann.gallery_ids))):
        raise ValueError("top1 must contain one valid gallery index per query")
    error_queries = np.flatnonzero(ann.gallery_ids[top1] != ann.query_ids)
    pred_indices = top1[error_queries]
    source_domains = ann.query_domains[error_queries]
    pred_domains = ann.gallery_domains[pred_indices]
    same_source = source_domains == pred_domains
    domain_to_index = {domain: index for index, domain in enumerate(DOMAIN_NAMES)}
    confusion = np.zeros((len(DOMAIN_NAMES), len(DOMAIN_NAMES)), dtype=np.int64)
    for source, predicted in zip(source_domains, pred_domains):
        confusion[domain_to_index[str(source)], domain_to_index[str(predicted)]] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float),
                           where=row_sums > 0)

    id_counts = Counter(ann.gallery_ids.tolist())
    domain_counts = Counter(ann.gallery_domains.tolist())
    id_domain_counts = Counter(zip(ann.gallery_ids.tolist(), ann.gallery_domains.tolist()))
    priors = np.empty(len(error_queries), dtype=np.float64)
    audit = []
    for offset, query_index in enumerate(error_queries):
        person_id = int(ann.query_ids[query_index])
        domain = str(ann.query_domains[query_index])
        eligible_count = len(ann.gallery_ids) - id_counts[person_id]
        same_domain_count = domain_counts[domain] - id_domain_counts[(person_id, domain)]
        if eligible_count <= 0:
            raise ValueError(f"query {query_index}: no gallery remains after excluding identity")
        priors[offset] = same_domain_count / eligible_count
        predicted = int(top1[query_index])
        audit.append(dict(
            query_index=int(query_index), source_image_index=int(ann.query_source_indices[query_index]),
            identity=person_id, view=int(ann.query_views[query_index]), source_domain=domain,
            predicted_image_index=predicted, predicted_domain=str(ann.gallery_domains[predicted]),
            predicted_view=int(ann.gallery_views[predicted]), same_source=bool(same_source[offset]),
            eligible_gallery_count=eligible_count, same_source_eligible_count=same_domain_count,
            conditional_prior=float(priors[offset]),
        ))

    groups = {}
    for view, name in ((0, "aerial"), (1, "ground")):
        selected = ann.query_views[error_queries] == view
        group_indicators, group_priors = same_source[selected], priors[selected]
        count = len(group_priors)
        same_probability = float(group_indicators.mean()) if count else None
        mean_prior = float(group_priors.mean()) if count else None
        enrichment = same_probability / mean_prior if count and mean_prior > 0 else None
        interval, invalid_replicates = paired_bootstrap_enrichment(group_indicators, group_priors)
        groups[name] = dict(
            n_queries=int((ann.query_views == view).sum()), n_errors=count,
            n_same_source_errors=int(group_indicators.sum()),
            same_source_error_probability=same_probability,
            mean_conditional_prior=mean_prior, enrichment_ratio=enrichment,
            enrichment_ci95=interval, bootstrap_repeats=BOOTSTRAP_REPEATS,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_zero_denominator_replicates=invalid_replicates,
            undefined_reason=("no error queries" if not count else
                              "zero conditional-prior denominator" if mean_prior == 0 else
                              "bootstrap has zero-denominator replicates" if invalid_replicates else None),
        )

    mixed_sources = {}
    same_view = ann.query_views[error_queries] == ann.gallery_views[pred_indices]
    for domain in ("AG-ReID.v2", "G2APS"):
        selected = source_domains == domain
        count = int(selected.sum())
        counts = dict(same_source_same_view=int((selected & same_source & same_view).sum()),
                      same_source_other_view=int((selected & same_source & ~same_view).sum()),
                      other_source=int((selected & ~same_source).sum()))
        mixed_sources[domain] = dict(
            n_errors=count, counts=counts,
            proportions={key: value / count if count else None for key, value in counts.items()},
        )
    return dict(
        annotation_summary=ann.summary, n_queries=len(top1), n_errors=len(error_queries),
        domains=list(DOMAIN_NAMES), confusion_counts=confusion.tolist(),
        confusion_row_normalized=[row.tolist() if total else [None] * len(DOMAIN_NAMES)
                                  for row, total in zip(normalized, row_sums[:, 0])],
        confusion_axes="rows = ground-truth source; columns = top-1 predicted source; errors only",
        enrichment_definition=(
            "P(predicted source equals true source | top-1 error, query view) / "
            "mean(same-source gallery fraction after removing ALL true-identity images "
            "| the same top-1 error queries and query view)"
        ),
        bootstrap_definition=(
            "10000 query-level paired resamples of (same-source-error indicator, conditional prior); "
            "numpy.default_rng(2026), separately for each view; percentile 95% interval; "
            "captions are sampling units, not identities or source images"
        ),
        groups=groups, mixed_sources=mixed_sources, error_query_audit=audit,
    )


def print_results(result):
    print("ERROR SOURCE CONFUSION (counts; rows=true, columns=predicted)")
    print_table(["true / pred", *DOMAIN_NAMES],
                [[domain, *row] for domain, row in zip(DOMAIN_NAMES, result["confusion_counts"])])
    print("ERROR SOURCE CONFUSION (row-normalized proportions; '-' = no errors)")
    print_table(["true / pred", *DOMAIN_NAMES],
                [[domain, *["-" if value is None else f"{value:.6f}" for value in row]]
                 for domain, row in zip(DOMAIN_NAMES, result["confusion_row_normalized"])])
    rows = []
    for name, group in result["groups"].items():
        interval = group["enrichment_ci95"]
        rows.append([name, group["n_errors"],
                     *["-" if group[key] is None else f"{group[key]:.6f}" for key in
                       ("same_source_error_probability", "mean_conditional_prior", "enrichment_ratio")],
                     "-" if interval is None else f"[{interval[0]:.6f}, {interval[1]:.6f}]"])
    print("CONDITIONAL ENRICHMENT (denominator: mean over the same error queries)")
    print_table(["query view", "errors", "P(same source)", "mean prior", "enrichment", "95% CI"], rows)
    print("MIXED-SOURCE ERROR PROPORTIONS")
    print_table(["source", "errors", "same source/view", "same source/other view", "other source"],
                [[domain, group["n_errors"],
                  *["-" if value is None else f"{value:.6f}" for value in group["proportions"].values()]]
                 for domain, group in result["mixed_sources"].items()])


def self_test():
    paths = ["AG-ReID.v2/aerial/0.jpg", "AG-ReID.v2/ground/1.jpg",
             "G2APS/aerial/2.jpg", "G2APS/ground/3.jpg", "Market/4.jpg", "Market/5.jpg",
             "PRAI-1581/6.jpg", "PRAI-1581/7.jpg", "uavhuman/8.jpg", "uavhuman/9.jpg"]
    ann = annotation_from_records(
        [dict(file_path=path, id=index, captions=["first caption", "second caption"])
         for index, path in enumerate(paths)], enforce_anchors=False
    )
    records = []
    for query_index, source_index in enumerate(ann.query_source_indices):
        predicted = int(source_index) ^ 1 if query_index % 2 == 0 else (int(source_index) + 2) % 10
        records.append({str(query_index): dict(pred=[predicted], match=[0])})
    result = analyze_errors(parse_predictions(records, ann), ann)
    assert result["n_queries"] == 20 and result["n_errors"] == 20
    assert np.asarray(result["confusion_counts"]).sum() == 20
    for group in result["groups"].values():
        assert group["same_source_error_probability"] == 0.5
        assert np.isclose(group["mean_conditional_prior"], 1 / 9)
        assert np.isclose(group["enrichment_ratio"], 4.5)
    assert all(np.isclose(row["conditional_prior"], 1 / 9) for row in result["error_query_audit"])
    assert all(group["proportions"] == dict(same_source_same_view=0.0,
                                            same_source_other_view=0.5, other_source=0.5)
               for group in result["mixed_sources"].values())
    empty = analyze_errors(ann.query_source_indices, ann)
    assert empty["n_errors"] == 0
    assert all(group["enrichment_ratio"] is None and group["enrichment_ci95"] is None
               for group in empty["groups"].values())
    # Separate multi-image-identity fixture verifies that *all* identity images,
    # not merely the query's source image, are excluded from each prior.
    multi = make_mock_annotation()
    wrong = np.asarray([np.flatnonzero(multi.gallery_ids != pid)[0] for pid in multi.query_ids])
    multi_result = analyze_errors(wrong, multi)
    for row in multi_result["error_query_audit"]:
        available = multi.gallery_ids != row["identity"]
        expected = (multi.gallery_domains[available] == row["source_domain"]).mean()
        assert row["eligible_gallery_count"] == int(available.sum())
        assert row["conditional_prior"] == expected
    assert multi_result["error_query_audit"][0]["eligible_gallery_count"] == 7
    assert np.array_equal(parse_predictions(records[::-1], ann), parse_predictions(records, ann))
    bad_records = [{**record} for record in records]
    bad_records[0] = {"0": dict(pred=[1], match=[1])}
    try:
        parse_predictions(bad_records, ann)
    except ValueError as error:
        print(f"EXPECTED SNAPSHOT VALIDATION ERROR: {error}")
    else:
        raise AssertionError("incorrect snapshot match flag was accepted")
    print_results(result)
    print("SELF_TEST PASS: 20 queries / 10 gallery, conditional prior=1/9, enrichment=4.5; "
          "10000 paired bootstrap replicates per view; all-identity exclusion, "
          "empty groups and invalid snapshot tested")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", help="output.json snapshot from the matching experiment")
    parser.add_argument("--ann", help="TAG-PEDES test_reid.json (published anchors are mandatory)")
    parser.add_argument("--out", help="output JSON, including per-error-query conditional priors")
    parser.add_argument("--self_test", action="store_true", help="run a 20-query/10-gallery synthetic test")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    missing = [name for name in ("pred", "ann", "out") if getattr(args, name) is None]
    if missing:
        parser.error("required arguments: " + ", ".join("--" + name for name in missing))
    try:
        ann = load_annotations(args.ann)
        with Path(args.pred).open(encoding="utf-8-sig") as handle:
            top1 = parse_predictions(json.load(handle), ann)
        result = analyze_errors(top1, ann)
        result["inputs"] = dict(pred=str(Path(args.pred).resolve()), ann=str(Path(args.ann).resolve()))
        print_results(result)
        write_json(args.out, result)
        print(f"Wrote {Path(args.out).resolve()}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
