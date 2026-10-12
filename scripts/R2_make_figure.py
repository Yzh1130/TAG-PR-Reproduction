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
import sys
import uuid

try:
    from .R2_hard_case_pool import output_directory, render_cases
    from ._analysis_audit import sha256_16
except ImportError:
    from R2_hard_case_pool import output_directory, render_cases
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--font", help="optional font file; no suitable font => text-free panels + layout JSON")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        try:
            from .R1_joint_enrichment import write_audited_report
        except ImportError:
            from R1_joint_enrichment import write_audited_report
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
        if args.font:
            inputs["font"] = args.font
        output = output_directory(args.out, inputs.values())
        run = output / ("R2_manual_figure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
        run.mkdir()
        conditions = list(dict.fromkeys(case["correction_condition"] for case in cases))
        rendered = render_cases(cases, run / "selected_panel.png", selection.get("title", "Manually selected cases"),
                                args.font, "; ".join(conditions))
        if rendered["font"]:
            inputs["render_font"] = rendered["font"]
        report = {"tool": "R2_make_figure", "status": "complete",
                  "manual_selection": selected, "selection_rule": "explicit --select human list; no metric-based selection",
                  "correction_conditions": conditions, "rendered": rendered, "image_input_manifest": manifest,
                  "text_fallback": "image-only panel and exact text/layout JSON if suitable Chinese font unavailable",
                  "readable_log": f"Manually selected {len(cases)} cases. conditions={conditions}; text_enabled={rendered['text_enabled']}"}
        write_audited_report(args, "R2_make_figure", report, inputs, {"random_seeds": [], "sampling": "none; manual selection only"}, argv)
        return 0
    except (OSError, ValueError, AssertionError, KeyError, TypeError) as exc:
        if "write_audited_report" in locals():
            try:
                write_audited_report(args, "R2_make_figure", {"status": "stopped", "error": str(exc)},
                                     inputs if "inputs" in locals() else {"selection": args.select},
                                     {"random_seeds": [], "sampling": "none; manual selection only"}, argv)
            except (OSError, ValueError) as report_error:
                print(f"Stopped-report write unavailable: {report_error}", file=sys.stderr)
        parser.exit(2, f"R2 figure input error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
