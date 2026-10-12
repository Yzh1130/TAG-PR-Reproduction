"""Render manually selected R2 cases; image-only fallback without a suitable font.

--select JSON schema: {"pool_json": "/absolute/R2_hard_case_pool.json",
"items": [{"pool": "transductive_success", "query_index": 123}, ...],
"title": "optional panel title"}. Cases must already be in the seeded pool.
The selection is explicit human input, not an automatic best-looking-case rule.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .R2_hard_case_pool import choose_font, output_directory, render_cases
    from ._analysis_audit import sha256_16
except ImportError:
    from R2_hard_case_pool import choose_font, output_directory, render_cases
    from _analysis_audit import sha256_16


def load_selection(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        selection = json.load(handle)
    if not isinstance(selection, dict) or not {"pool_json", "items"} <= selection.keys():
        raise ValueError("selection JSON requires pool_json and items")
    pool_path = Path(selection["pool_json"]).expanduser()
    if not pool_path.is_absolute():
        pool_path = Path(path).resolve().parent / pool_path
    with pool_path.open(encoding="utf-8-sig") as handle:
        pool = json.load(handle)
    if not isinstance(selection["items"], list) or not selection["items"]:
        raise ValueError("selection items must be a nonempty list")
    seen, cases, selected = set(), [], []
    for item in selection["items"]:
        if not isinstance(item, dict) or not {"pool", "query_index"} <= item.keys():
            raise ValueError("each selection item requires pool and query_index")
        if type(item["query_index"]) is not int:
            raise ValueError("query_index must be an integer")
        key = (item["pool"], item["query_index"])
        if key in seen:
            raise ValueError(f"duplicate manual selection: {key}")
        seen.add(key)
        if key[0] not in pool["pools"]:
            raise ValueError(f"unknown candidate pool: {key[0]}")
        matching = [case for case in pool["pools"][key[0]]["cases"] if case["query_index"] == key[1]]
        if len(matching) != 1:
            raise ValueError(f"selected query is not in saved seeded pool: {key}")
        cases.append(matching[0])
        selected.append({**item, "correction_condition": matching[0]["correction_condition"],
                         "occlusion_manual_confirmation_required": bool(pool["pools"][key[0]].get("manual_confirmation_required")),
                         "human_confirmation": item.get("human_confirmation")})
    return selection, pool_path.resolve(), cases, selected


def load_panels(panel_specs, pool_specs, select_path=None):
    """Resolve explicit saved rows only; neither matrices nor candidates are reconstructed.

    @label selects a JSON, not a correction mode. /pool optionally disambiguates
    different correction conditions saved for one query within that JSON.
    """
    sources = {}
    for spec in pool_specs:
        if "=" not in spec:
            raise ValueError("--pool-json requires LABEL=PATH")
        label, value = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or not value or label in sources:
            raise ValueError("pool JSON labels must be nonempty, unique ASCII names")
        sources[label] = Path(value).expanduser().resolve()
    if select_path:
        with Path(select_path).open(encoding="utf-8-sig") as handle:
            selection = json.load(handle)
        value = selection.get("pool_json") if isinstance(selection, dict) else None
        if value:
            source = Path(value).expanduser()
            if not source.is_absolute():
                source = Path(select_path).resolve().parent / source
        elif isinstance(selection, dict) and "pools" in selection:
            source = Path(select_path)
        else:
            raise ValueError("--select with --panel requires a pool JSON or a legacy selection JSON referencing one")
        if "default" in sources and sources["default"] != source.resolve():
            raise ValueError("--select conflicts with --pool-json default=PATH")
        sources["default"] = source.resolve()
    if not sources:
        raise ValueError("--panel requires --pool-json LABEL=PATH or --select POOL_JSON")
    documents, manifest = {}, {}
    for label, path in sources.items():
        with path.open(encoding="utf-8-sig") as handle:
            document = json.load(handle)
        if not isinstance(document, dict) or not isinstance(document.get("pools"), dict):
            raise ValueError(f"{label}: input is not an R2 pool JSON")
        documents[label] = document
        manifest[label] = {"path": str(path), "bytes": path.stat().st_size, "sha256_16": sha256_16(path)}
    cases, selected, seen = [], [], set()
    for spec in panel_specs:
        if ":" not in spec:
            raise ValueError("--panel requires LETTER:QUERY[,QUERY...] (optionally QUERY@LABEL/POOL)")
        letter, tokens = spec.split(":", 1)
        if not re.fullmatch(r"[A-Za-z]", letter) or not tokens:
            raise ValueError("panel labels must be single ASCII letters and contain query indices")
        for token in tokens.split(","):
            match = re.fullmatch(r"(\d+)(?:@([A-Za-z0-9_-]+)(?:/([A-Za-z0-9_-]+))?)?", token)
            if not match:
                raise ValueError(f"invalid panel query specification: {token!r}")
            query, label, requested_pool = int(match[1]), match[2], match[3]
            if label is not None and label not in documents:
                raise ValueError(f"unknown pool JSON label: {label}")
            labels = [label] if label else list(documents)
            matches, candidate_only = [], []
            for origin in labels:
                pools = documents[origin]["pools"]
                if requested_pool is not None and requested_pool not in pools:
                    raise ValueError(f"{origin}: unknown saved pool {requested_pool}")
                for pool_name, pool in pools.items():
                    if requested_pool is not None and requested_pool != pool_name:
                        continue
                    saved = [case for case in pool.get("cases", []) if case.get("query_index") == query]
                    if len(saved) > 1:
                        raise ValueError(f"duplicate saved rows for q={query} in {origin}/{pool_name}")
                    for case in saved:
                        condition = case.get("correction_condition", pool.get("correction_condition"))
                        if not isinstance(condition, str):
                            raise ValueError(f"q={query}: saved row has no correction condition")
                        sampled = pool.get("selected_query_indices")
                        matches.append((case, {"pool": pool_name, "pool_json_label": origin,
                            "pool_sampling_seed": pool.get("seed"),
                            "in_seeded_30_sample": (query in sampled) if isinstance(sampled, list) else None,
                            "saved_sample_size": len(sampled) if isinstance(sampled, list) else None,
                            "sample_membership_basis": "membership in saved selected_query_indices; sample size is recorded, not assumed to be 30",
                            "correction_condition": condition, "source_json": manifest[origin]["path"],
                            "source_json_sha256_16": manifest[origin]["sha256_16"],
                            "source_json_bytes": manifest[origin]["bytes"]}))
                    if not saved and query in pool.get("candidate_query_indices", []):
                        candidate_only.append(f"{origin}/{pool_name}")
            if not matches:
                suffix = f"; candidate only in {candidate_only}, with no saved image/top-3 row" if candidate_only else ""
                raise ValueError(f"q={query} is missing from saved cases{suffix}; cannot reconstruct or silently skip")
            conditions = {entry[1]["correction_condition"] for entry in matches}
            if len(conditions) != 1:
                raise ValueError(f"q={query} has different correction conditions; specify @LABEL and, if needed, /POOL_NAME")
            reference = matches[0][0]
            # Same condition is not enough if artifacts disagree about retrieved images.
            for case, _ in matches[1:]:
                if case != reference:
                    raise ValueError(f"q={query}: same condition has inconsistent saved rows; select @LABEL/POOL_NAME")
            key = (letter.lower(), query, next(iter(conditions)))
            if key in seen:
                raise ValueError(f"duplicate panel row: {spec}")
            seen.add(key)
            cases.append(reference)
            selected.append({"panel": letter.lower(), "query_index": query,
                "correction_condition": next(iter(conditions)), "pool_memberships": [item[1] for item in matches],
                "query_text": reference.get("query_text"),
                "occlusion_manual_confirmation_required": any(documents[item[1]["pool_json_label"]]["pools"][item[1]["pool"]].get("manual_confirmation_required", False) for item in matches)})
    return sources, cases, selected


ENGLISH_PANEL_HEADERS = ["Query source", "Before #1", "Before #2", "Before #3",
                         "After #1", "After #2", "After #3", "Best positive"]
CHINESE_PANEL_HEADERS = ["查询源图", "校正前#1", "校正前#2", "校正前#3",
                         "校正后#1", "校正后#2", "校正后#3", "最优正样本"]


def _chinese_panel_font(font_path):
    """Use only the requested font for Chinese; never probe another CJK font."""
    reason = None
    if not font_path:
        reason = "--labels zh requires an explicit --font with all header glyphs"
    else:
        try:
            # choose_font checks every required CJK glyph against the font's
            # replacement-glyph mask, rather than merely accepting a file path.
            font, font_name = choose_font(CHINESE_PANEL_HEADERS, font_path, size=15)
            if font is not None:
                return font, font_name, None
            reason = "requested font is unusable or lacks required Chinese header glyphs"
        except (OSError, ValueError) as exc:
            reason = str(exc)
    print(f"WARNING: {reason}; falling back to English panel labels", flush=True)
    # English fallback may use the requested font if it can be loaded. It must
    # not silently substitute a different Chinese font for the requested one.
    try:
        font, font_name = choose_font(ENGLISH_PANEL_HEADERS, font_path, size=15)
    except (OSError, ValueError):
        font, font_name = None, None
    if font is None:
        font, font_name = choose_font(ENGLISH_PANEL_HEADERS, size=15)
    if font is None:
        font = ImageFont.load_default()
    return font, font_name, reason


def render_panels(cases, selected, path, font_path=None, labels="en"):
    """Publication panels: optional Chinese headers; query text lives in JSON.

    Uses the legacy eight-image row layout, with a small panel/q label gutter.
    No textual data from annotations or free-form provenance is drawn in the image.
    """
    cell_width, image_height, gutter, header, row_height = 180, 200, 110, 35, 245
    if labels not in ("en", "zh"):
        raise ValueError("panel labels must be en or zh")
    fallback = None
    if labels == "zh":
        font, font_name, fallback = _chinese_panel_font(font_path)
    else:
        # Keep the original English font resolution and drawing path unchanged.
        font, font_name = choose_font(["Query Before After Best positive"], font_path, size=15)
        if font is None:
            print("WARNING: no suitable font; using the built-in English font for panel labels")
            font = ImageFont.load_default()
    rendered_labels = "zh" if labels == "zh" and fallback is None else "en"
    canvas = Image.new("RGB", (gutter + cell_width * 8, header + row_height * len(cases)), "white")
    draw = ImageDraw.Draw(canvas)
    layout = {"query_text_drawn": False, "language": "Chinese" if rendered_labels == "zh" else "English", "dpi": 300,
              "font": font_name, "canvas_size": list(canvas.size), "rows": []}
    roles = CHINESE_PANEL_HEADERS if rendered_labels == "zh" else ENGLISH_PANEL_HEADERS
    label_metadata = ({"requested_labels": labels, "rendered_labels": rendered_labels,
                       "column_headers": roles, "label_font_fallback": fallback} if labels == "zh" else {})
    layout.update(label_metadata)
    for col, role in enumerate(roles):
        draw.text((gutter + col * cell_width + 5, 5), role, fill="black", font=font)
    for row, (case, selection) in enumerate(zip(cases, selected)):
        top = header + row * row_height
        draw.text((5, top + 5), f"({selection['panel']})\nq={case['query_index']}", fill="black", font=font, spacing=5)
        slots = [{"role": "query_source", **case["source_image"]},
                 *[{"role": f"before_rank_{i + 1}", **item} for i, item in enumerate(case["before_top3"])],
                 *[{"role": f"after_rank_{i + 1}", **item} for i, item in enumerate(case["after_top3"])],
                 {"role": "best_positive", **case["truth_image"]}]
        if len(slots) != 8:
            raise ValueError(f"q={case['query_index']}: expected source + top-3 + top-3 + best-positive images")
        row_layout = {**selection, "slots": []}
        for col, slot in enumerate(slots):
            with Image.open(slot["path"]) as source:
                image = ImageOps.contain(source.convert("RGB"), (cell_width - 10, image_height - 10))
            left = gutter + col * cell_width
            canvas.paste(image, (left + (cell_width - image.width) // 2, top + (image_height - image.height) // 2))
            rectangle = [left, top, left + cell_width - 1, top + image_height - 1]
            color = "green" if slot.get("hit") else "red"
            draw.rectangle(rectangle, outline=color, width=3)
            draw.text((left + 5, top + image_height + 5), f"g={slot['gallery_index']} pid={slot['pid']}", fill="black", font=font)
            row_layout["slots"].append({**slot, "image_rectangle": rectangle, "frame_color": color})
        layout["rows"].append(row_layout)
    path = Path(path)
    with path.open("xb") as handle:
        canvas.save(handle, format="PNG", dpi=(300, 300))
    pdf = path.with_suffix(".pdf")
    with pdf.open("xb") as handle:
        canvas.save(handle, format="PDF", resolution=300)
    layout_path = path.with_suffix(".layout.json")
    with layout_path.open("x", encoding="utf-8") as handle:
        json.dump(layout, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"png": str(path.resolve()), "pdf": str(pdf.resolve()), "layout_json": str(layout_path.resolve()),
            "font": font_name, "n_rows": len(cases), "query_text_drawn": False, "dpi": 300, **label_metadata}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", help="legacy manual selection JSON; with --panel may also reference an R2 pool JSON")
    parser.add_argument("--pool-json", action="append", default=[], help="repeatable LABEL=R2_JSON for explicit --panel rows")
    parser.add_argument("--panel", action="append", default=[], help="repeatable LETTER:QUERY[,QUERY...] with optional @LABEL/POOL_NAME")
    parser.add_argument("--tag", default="selected", help="explicit-panel output tag (default: selected)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--font", help="optional font file; required for Chinese panel headers; legacy sheets may fall back to text-free; query text is never drawn in explicit panels")
    parser.add_argument("--labels", choices=("en", "zh"), default="en",
                        help="explicit --panel header language (default: en); zh falls back to en with a warning if --font is unavailable or lacks required glyphs")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = {**({"selection": args.select} if args.select else {}),
              **{f"pool_json_arg_{i}": spec.split("=", 1)[1] for i, spec in enumerate(args.pool_json) if "=" in spec}}
    try:
        try:
            from .R1_joint_enrichment import write_audited_report
        except ImportError:
            from R1_joint_enrichment import write_audited_report
        if args.panel:
            sources, cases, selected = load_panels(args.panel, args.pool_json, args.select)
            inputs = {f"pool_json_{label}": path for label, path in sources.items()}
            if args.select:
                inputs["selection"] = args.select
            selection = {}
            if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
                raise ValueError("--tag must contain only ASCII letters, digits, underscore or hyphen")
        else:
            if args.labels != "en":
                raise ValueError("--labels zh requires explicit --panel rows")
            if not args.select:
                raise ValueError("supply legacy --select JSON or explicit --panel with --pool-json")
            if args.pool_json:
                raise ValueError("--pool-json requires --panel")
            selection, pool_path, cases, selected = load_selection(args.select)
            inputs = {"selection": args.select, "pool_json": pool_path}
        manifest = []
        for case in cases:
            images = [case["source_image"], *case["before_top3"], *case["after_top3"], case["truth_image"]]
            for image in images:
                path = Path(image["path"]).resolve()
                role = "image_" + str(image["gallery_index"])
                if role in inputs and inputs[role] != path:
                    raise ValueError("one gallery index maps to inconsistent image paths")
                if role not in inputs:
                    inputs[role] = path
                    manifest.append({"gallery_index": image["gallery_index"], "path": str(path),
                                     "sha256_16": sha256_16(path), "bytes": path.stat().st_size})
        if args.font and (args.labels == "en" or Path(args.font).is_file()):
            inputs["font"] = args.font
        output = output_directory(args.out, inputs.values())
        run = output / ("R2_manual_figure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
        run.mkdir()
        conditions = list(dict.fromkeys(case["correction_condition"] for case in cases))
        if args.panel:
            rendered = render_panels(cases, selected, run / f"selected_panel_{args.tag}.png", args.font, args.labels)
        else:
            rendered = render_cases(cases, run / "selected_panel.png", selection.get("title", "Manually selected cases"),
                                    args.font, "; ".join(conditions))
        if rendered["font"]:
            inputs["render_font"] = rendered["font"]
        report = {"tool": "R2_make_figure", "status": "complete",
                  "manual_selection": selected, "selection_rule": ("explicit --panel saved-case list; no metric-based selection" if args.panel
                                                                       else "explicit --select human list; no metric-based selection"),
                  "correction_conditions": conditions, "rendered": rendered, "image_input_manifest": manifest,
                  "text_fallback": (("explicit panels use the requested Chinese header font or warn and fall back to English; query text only in JSON" if args.labels == "zh"
                                     else "explicit panels draw English labels with a built-in font fallback; query text only in JSON") if args.panel
                                    else "image-only panel and exact text/layout JSON if suitable Chinese font unavailable"),
                  "readable_log": (f"Manually selected {len(cases)} cases. conditions={conditions}; query_text_drawn=False" if args.panel
                                   else f"Manually selected {len(cases)} cases. conditions={conditions}; text_enabled={rendered['text_enabled']}")}
        if args.panel:
            from argparse import Namespace
            selected_out = Namespace(**vars(args))
            selected_out.out = str(run / f"selected_cases_{args.tag}.json")
            selected_report = {"tool": "R2_make_figure", "status": "complete", "manual_selection": selected,
                               "cases": cases, "rendered": rendered, "image_input_manifest": manifest,
                               "selection_rule": "explicit saved cases only; no matrix extraction or automatic selection"}
            write_audited_report(selected_out, "R2_make_figure", selected_report, inputs,
                                 {"random_seeds": [], "sampling": "none; manual selection only",
                                  "origin_pool_seeds": [membership["pool_sampling_seed"] for row in selected for membership in row["pool_memberships"]]}, argv)
            report["selected_cases_json"] = selected_out.out
        write_audited_report(args, "R2_make_figure", report, inputs, {"random_seeds": [], "sampling": "none; manual selection only"}, argv)
        return 0
    except (OSError, ValueError, AssertionError, KeyError, TypeError) as exc:
        if "write_audited_report" in locals():
            try:
                write_audited_report(args, "R2_make_figure", {"status": "stopped", "error": str(exc)},
                                     inputs,
                                     {"random_seeds": [], "sampling": "none; manual selection only"}, argv)
            except (OSError, ValueError) as report_error:
                print(f"Stopped-report write unavailable: {report_error}", file=sys.stderr)
        parser.exit(2, f"R2 figure input error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
