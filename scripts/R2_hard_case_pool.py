"""Read-only, seeded hard-case pools and contact sheets for manual case selection.

Occlusion keyword matches are candidates only, explicitly awaiting manual review.
No training, metric definitions, feature extraction or existing artifacts change.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import struct
import sys
import textwrap
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from ._offline_common import compact_metrics, evaluate, load_annotations, load_matrix, load_similarity, safe_output_dir, topk_mean
    from .R3_finegrained_causes import ATTRIBUTE_VOCABULARY, attribute_comparison, image_path, load_records, read_image_sizes
except ImportError:
    from _offline_common import compact_metrics, evaluate, load_annotations, load_matrix, load_similarity, safe_output_dir, topk_mean
    from R3_finegrained_causes import ATTRIBUTE_VOCABULARY, attribute_comparison, image_path, load_records, read_image_sizes


OCCLUSION_KEYWORDS = ["occlu", "partial", "block", "behind", "umbrella", "carry"]
POOL_NAMES = ("small_target", "similar_clothing", "occlusion_manual_review",
              "transductive_success", "transductive_failure", "inductive_success", "inductive_failure")


def output_directory(out, inputs):
    target = Path(out).expanduser().resolve()
    directory = target.parent if target.suffix.lower() == ".json" else target
    if any(part.lower() in ("_archive_evidence", "_dump") for part in directory.parts):
        raise ValueError("outputs must be outside immutable evidence archives")
    for source in inputs:
        parent = Path(source).expanduser().resolve().parent
        if parent == directory or parent in directory.parents:
            raise ValueError(f"output must be outside input directory: {parent}")
    return safe_output_dir(directory)


def choose_font(texts, font_path=None, size=15):
    needs_cjk = any(re.search(r"[\u3400-\u9fff]", str(text)) for text in texts)
    cjk_candidates = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
    if font_path:
        if not Path(font_path).is_file():
            raise ValueError(f"font file does not exist: {font_path}")
        candidates = [str(font_path)]
        # A user-supplied font is explicit; verify CJK glyphs against the known
        # replacement glyph before using it for Chinese text.
    else:
        candidates = cjk_candidates if needs_cjk else ["C:/Windows/Fonts/arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", *cjk_candidates]
    for path in candidates:
        if not Path(path).is_file():
            continue
        try:
            font = ImageFont.truetype(path, size)
            if needs_cjk:
                characters = set(re.findall(r"[\u3400-\u9fff]", " ".join(str(x) for x in texts)))
                replacement = bytes(font.getmask("\U0010ffff"))
                if any(bytes(font.getmask(char)) == replacement for char in characters):
                    continue
            return font, str(Path(path).resolve())
        except OSError:
            continue
    return None, None


def _save_png_exclusive(image, path):
    with Path(path).open("xb") as handle:
        image.save(handle, format="PNG")


def render_cases(cases, path, title="", font_path=None, condition=""):
    """Eight columns: source, before top3, after top3, best positive image."""
    cell_width, image_height = 180, 200
    captions = [case["query_text"] for case in cases]
    font, selected_font = choose_font([title, condition, *captions], font_path)
    text_enabled = font is not None
    if not text_enabled:
        print("WARNING: no suitable font; writing image-only panel and layout JSON for paper-side text composition")
    header = 50 if text_enabled else 0
    query_lines = [textwrap.wrap(f"q={case['query_index']} {case['query_text']}", 170) for case in cases]
    row_heights = [image_height + (max(130, 75 + 18 * len(lines)) if text_enabled else 0)
                   for lines in query_lines]
    canvas = Image.new("RGB", (cell_width * 8, max(header + sum(row_heights), 1)), "white")
    draw = ImageDraw.Draw(canvas)
    if text_enabled:
        draw.text((8, 5), title, fill="black", font=font)
        draw.text((8, 25), condition, fill="black", font=font)
    layout = {"text_enabled": text_enabled, "font": selected_font, "title": title,
              "correction_condition": condition, "canvas_size": list(canvas.size), "rows": []}
    top = header
    for row_index, case in enumerate(cases):
        slots = [{"role": "query_source", **case["source_image"]},
                 *[{"role": f"before_rank_{i + 1}", **item} for i, item in enumerate(case["before_top3"])],
                 *[{"role": f"after_rank_{i + 1}", **item} for i, item in enumerate(case["after_top3"])],
                 {"role": "best_positive", **case["truth_image"]}]
        row_layout = {"query_index": case["query_index"], "query_text": case["query_text"],
                      "pid": case["pid"], "before_correct": case["before_correct"],
                      "after_correct": case["after_correct"], "slots": []}
        for col, slot in enumerate(slots):
            with Image.open(slot["path"]) as source:
                image = ImageOps.contain(source.convert("RGB"), (cell_width - 10, image_height - 10))
            x, y = col * cell_width + (cell_width - image.width) // 2, top + (image_height - image.height) // 2
            canvas.paste(image, (x, y))
            rectangle = [col * cell_width, top, (col + 1) * cell_width - 1, top + image_height - 1]
            color = "green" if slot.get("hit") else "red" if slot["role"].startswith(("before", "after")) else "blue"
            draw.rectangle(rectangle, outline=color, width=3)
            label = f"{slot['role']} g={slot['gallery_index']} pid={slot['pid']} hit={slot.get('hit')}"
            if text_enabled:
                draw.multiline_text((col * cell_width + 5, top + image_height + 3),
                                    "\n".join(textwrap.wrap(label, 23)), fill="black", font=font)
            row_layout["slots"].append({**slot, "image_rectangle": rectangle, "label": label})
        if text_enabled:
            draw.multiline_text((5, top + image_height + 55),
                                "\n".join(query_lines[row_index]),
                                fill="black", font=font)
        layout["rows"].append(row_layout)
        top += row_heights[row_index]
    _save_png_exclusive(canvas, path)
    layout_path = Path(path).with_suffix(".layout.json")
    with layout_path.open("x", encoding="utf-8") as handle:
        json.dump(layout, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"png": str(Path(path).resolve()), "layout_json": str(layout_path.resolve()),
            "text_enabled": text_enabled, "font": selected_font, "n_rows": len(cases)}


def _rank3(row):
    return np.argsort(-np.asarray(row), kind="stable")[:3]


def make_case(query, sim, ann, texts, root, penalty, condition, alpha=.5):
    before = np.asarray(sim[query], dtype=np.float32)
    after = before - np.float32(alpha) * penalty
    predicted_before, predicted_after = _rank3(before), _rank3(after)
    positives = np.flatnonzero(ann.gallery_ids == ann.query_ids[query])
    best_positive = int(positives[np.argmax(before[positives])])
    pid = int(ann.query_ids[query])

    def image_info(index):
        return {"gallery_index": int(index), "path": str(image_path(root, ann.paths[index])),
                "pid": int(ann.gallery_ids[index]), "view": int(ann.gallery_views[index]),
                "source": str(ann.gallery_domains[index]), "hit": bool(ann.gallery_ids[index] == pid)}

    return {"query_index": int(query), "query_text": texts[query], "pid": pid,
            "query_view": int(ann.query_views[query]), "query_source": str(ann.query_domains[query]),
            "source_image": image_info(int(ann.query_source_indices[query])),
            "before_top3": [{**image_info(index), "score": float(before[index])} for index in predicted_before],
            "after_top3": [{**image_info(index), "score": float(after[index])} for index in predicted_after],
            "truth_image": image_info(best_positive), "truth_gallery_indices": positives.tolist(),
            "before_correct": bool(ann.gallery_ids[predicted_before[0]] == pid),
            "after_correct": bool(ann.gallery_ids[predicted_after[0]] == pid),
            "correction_condition": condition}


def analyze(sim, bank, ann, records, areas, root, s13, seed=2026, cases_per_pool=30,
            inductive_alpha=.5, inductive_k=10, bank_label=None, bank_info=None):
    if seed < 0 or cases_per_pool < 1:
        raise ValueError("seed must be nonnegative; cases per pool positive")
    if not np.isfinite(inductive_alpha) or inductive_alpha < 0 or type(inductive_k) is not int or inductive_k < 1:
        raise ValueError("inductive alpha must be finite/nonnegative and k a positive integer")
    if bank.ndim != 2 or bank.shape[1] != sim.shape[1] or bank.shape[0] < inductive_k:
        raise ValueError(f"bank needs >={inductive_k} rows and the same gallery columns as sim")
    raw = evaluate(sim, ann)
    transductive_penalty, inductive_penalty = topk_mean(sim, 10), topk_mean(bank, inductive_k)
    transductive = evaluate(sim, ann, penalty=transductive_penalty, alpha=.5)
    inductive = evaluate(sim, ann, penalty=inductive_penalty, alpha=inductive_alpha)
    c = s13["configurations"]["c"]
    if (c["bank"], c["alpha"], c["k"]) != ("test_full", .5, 10):
        raise ValueError("S13 configuration c must be test_full alpha=.5 k=10")
    expected = c["metrics"]["R1"]
    if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not np.isfinite(expected):
        raise ValueError("S13 configuration c metrics.R1 must be a finite number")
    if struct.pack("!d", transductive["R1"]) != struct.pack("!d", expected):
        raise AssertionError(f"S13 c corrected R1 exact check FAILED: {transductive['R1']!r} != {expected!r}")
    if struct.pack("!d", raw["R1"]) != struct.pack("!d", s13["raw"]["R1"]):
        raise AssertionError("S13 uncorrected R1 exact check FAILED")
    correct, trans_correct, ind_correct = (result["details"]["correct"] for result in (raw, transductive, inductive))
    attrs = attribute_comparison(records, ann, raw["details"]["top1"], correct, seed)
    errors, overlap = attrs["error_queries"], attrs["error_jaccard"]
    area_cut = float(np.quantile(areas, .1))
    overlap_cut = float(np.quantile(overlap, .9)) if len(overlap) else None
    occlusion = np.asarray([any(word in text.lower() for word in OCCLUSION_KEYWORDS)
                            for text in attrs["texts"]], dtype=bool)
    candidate_sets = {
        "small_target": np.flatnonzero((np.asarray(areas)[ann.query_source_indices] <= area_cut) & ~correct),
        "similar_clothing": errors[overlap >= overlap_cut] if overlap_cut is not None else np.empty(0, np.int64),
        "occlusion_manual_review": np.flatnonzero(occlusion & ~correct),
        "transductive_success": np.flatnonzero(~correct & trans_correct),
        "transductive_failure": np.flatnonzero(correct & ~trans_correct),
        "inductive_success": np.flatnonzero(~correct & ind_correct),
        "inductive_failure": np.flatnonzero(correct & ~ind_correct),
    }
    pools = {}
    for offset, name in enumerate(POOL_NAMES):
        candidates = candidate_sets[name]
        pool_seed = seed + offset
        rng = np.random.default_rng(pool_seed)
        selected = np.sort(rng.choice(candidates, size=min(cases_per_pool, len(candidates)), replace=False))
        is_inductive = name.startswith("inductive")
        penalty = inductive_penalty if is_inductive else transductive_penalty
        condition = (("training-text bank CSLS alpha=.5 k=10" if inductive_alpha == .5 and inductive_k == 10
                      else f"training-text bank CSLS alpha={inductive_alpha:g} k={inductive_k}")
                     if is_inductive else "complete-test-query transductive CSLS alpha=.5 k=10")
        cases = [make_case(int(q), sim, ann, attrs["texts"], root, penalty, condition,
                           inductive_alpha if is_inductive else .5) for q in selected]
        pools[name] = {"candidate_count": len(candidates), "candidate_query_indices": candidates.tolist(),
                       "selected_count": len(selected), "seed": pool_seed,
                       "selected_query_indices": selected.tolist(), "cases": cases,
                       "manual_confirmation_required": name == "occlusion_manual_review",
                       "correction_condition": condition}
    return {"tool": "R2_hard_case_pool", "status": "complete",
            "raw": compact_metrics(raw), "transductive": compact_metrics(transductive),
            "inductive": {**compact_metrics(inductive), "alpha": inductive_alpha, "k": inductive_k,
                          "bank": bank_info, "bank_label": bank_label}, "annotation": ann.summary,
            "training_bank_provenance_note": (f"User-supplied bank label: {bank_label}. "
                "This label alone does not certify query independence; retain the supplying extraction JSON."
                if bank_label is not None else s13.get("training_bank_provenance_note",
                "A training-bank label alone does not certify query independence; retain the supplying S12/E3 extraction JSON.")),
            "inductive_label_scope": "training-text-bank correction; an S12 test-fitted W bank remains an offline simulation",
            "parity_gate": {"S13_configuration": "c", "passed_exact_R1": True, "expected_R1": expected,
                            "actual_R1": transductive["R1"], "raw_exact_R1": True},
            "rules": {"small_target": "source-image area <= empirical gallery-area 10th percentile AND raw top1 error; ties retained",
                      "small_target_area_cut_pixels2": area_cut,
                      "similar_clothing": "top 10% error-image max-caption attribute Jaccard among raw errors; threshold ties retained",
                      "similar_clothing_jaccard_cut": overlap_cut,
                      "occlusion": "case-insensitive substring match AND raw top1 error; manual confirmation required",
                      "occlusion_keywords": OCCLUSION_KEYWORDS,
                      "success": "raw wrong, specified correction correct", "failure": "raw correct, specified correction wrong",
                      "selection": "each pool seed = --seed + fixed pool offset; sample without replacement; at most requested count; sorted selected query indices",
                      "source_exclusion": False},
            "attribute_vocabulary": ATTRIBUTE_VOCABULARY,
            "pools": pools}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sim", "bank", "anno", "image-root", "s13-json", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cases-per-pool", type=int, default=30)
    parser.add_argument("--sheet-page-size", type=int, default=6)
    parser.add_argument("--font", help="optional explicit font file; otherwise probe installed fonts")
    parser.add_argument("--inductive-alpha", type=float, default=.5)
    parser.add_argument("--inductive-k", type=int, default=10)
    parser.add_argument("--bank-label", default=None, help="user-supplied provenance label, not a certification of query independence")
    return parser


def main(argv=None):
    parser, args = build_parser(), None
    args = parser.parse_args(argv)
    if args.seed < 0 or min(args.cases_per_pool, args.sheet_page_size) < 1:
        parser.error("--seed must be nonnegative; pool/page sizes positive")
    try:
        try:
            from .R1_joint_enrichment import write_audited_report
        except ImportError:
            from R1_joint_enrichment import write_audited_report
        output = output_directory(args.out, [args.sim, args.bank, args.anno, args.s13_json])
        ann, records = load_annotations(args.anno), load_records(args.anno)
        sim, bank = load_similarity(args.sim, ann), load_matrix(args.bank)
        with Path(args.s13_json).open(encoding="utf-8-sig") as handle:
            s13 = json.load(handle)
        _, areas, manifest = read_image_sizes(ann, args.image_root)
        try:
            from ._analysis_audit import sha256_16
        except ImportError:
            from _analysis_audit import sha256_16
        bank_path = Path(args.bank).expanduser().resolve()
        bank_info = {"path": str(bank_path), "bytes": bank_path.stat().st_size,
                     "sha256_16": sha256_16(bank_path)}
        report = analyze(sim, bank, ann, records, areas, args.image_root, s13, args.seed, args.cases_per_pool,
                         args.inductive_alpha, args.inductive_k, args.bank_label, bank_info)
        run = output / ("R2_case_sheets_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
        run.mkdir()
        for name, pool in report["pools"].items():
            pool["contact_sheets"] = [render_cases(pool["cases"][start:start + args.sheet_page_size],
                                     run / f"{name}_{start // args.sheet_page_size:02d}.png", name, args.font,
                                     pool["correction_condition"])
                                     for start in range(0, len(pool["cases"]), args.sheet_page_size)]
            print(f"{name}: candidates={pool['candidate_count']} selected={pool['selected_count']}")
        report["image_input_manifest"] = manifest
        report["image_manifest_provenance_note"] = "Each original input image sha256_16 and bytes recorded in image_input_manifest."
        inputs = {"sim": args.sim, "bank": args.bank, "anno": args.anno, "s13_json": args.s13_json,
                  **({"font": args.font} if args.font else {})}
        used_fonts = sorted({sheet["font"] for entry in report["pools"].values()
                             for sheet in entry["contact_sheets"] if sheet["font"]})
        inputs.update({f"render_font_{index}": font for index, font in enumerate(used_fonts)})
        path = write_audited_report(args, "R2_hard_case_pool", report,
                                   inputs,
                                   {"attribute_random_negative": args.seed, "pool_seeds": {name: args.seed + i for i, name in enumerate(POOL_NAMES)}}, argv)
        print("R2 case pools complete:", path)
        return 0
    except (OSError, ValueError, AssertionError, KeyError, TypeError, struct.error) as exc:
        if "write_audited_report" in locals():
            try:
                write_audited_report(args, "R2_hard_case_pool",
                                     {"status": "stopped", "error": str(exc)},
                                     {"sim": args.sim, "bank": args.bank, "anno": args.anno, "s13_json": args.s13_json},
                                     {"attribute_random_negative": args.seed,
                                      "pool_seeds": {name: args.seed + i for i, name in enumerate(POOL_NAMES)}}, argv)
            except (OSError, ValueError) as report_error:
                print(f"Stopped-report write unavailable: {report_error}", file=sys.stderr)
        parser.exit(2, f"R2 input/self-check error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
