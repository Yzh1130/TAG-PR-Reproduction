"""200 x 100 external-run fixtures against the checked-in original S7 code.

No retrieval metric is hard-coded. Original and parameterized programs consume
identical synthetic inputs. Only fixture cardinalities (4220->80, 3200->200)
and iteration budgets are substituted in BOTH executions, not production code.
"""

import ast
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "scripts" / "ext"
REFERENCE = Path(__file__).parent / "reference_ext"
NAMES = ("S7a_csls", "S7b_distractor", "S7c_valselect", "S7d_aeri", "S7e_tbapr")
HELPERS = importlib.import_module("scripts.ext.S7_aggregate_seeds")


def normalized(rng, rows, width):
    result = rng.normal(size=(rows, width)).astype(np.float32)
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, seed=0):
    rng = np.random.default_rng(2026 + seed)
    paths = {}
    for dataset, split in [(d, "test") for d in HELPERS.DATASETS] + [("cuhkpedes", "val")]:
        path = root / "runs" / f"{dataset}_ViT-B-16_{split}_seed{seed}_mock"
        path.mkdir(parents=True)
        ig, il = normalized(rng, 100, 8), normalized(rng, 100, 16)
        source = np.repeat(np.arange(100), 2)  # caption expansion; AG rows then G2 rows
        tg = normalized(rng, 200, 8) + ig[source] * .25
        tl = normalized(rng, 200, 16) + il[source] * .25
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        tl /= np.linalg.norm(tl, axis=1, keepdims=True)
        arrays = {"feats_img_global": ig, "feats_img_local": il,
                  "feats_text_global": tg, "feats_text_local": tl,
                  "feats_bank_text_global": normalized(rng, 300, 8),
                  "feats_bank_text_local": normalized(rng, 300, 16),
                  "feats_extra_img_global": normalized(rng, 80, 8),
                  "feats_extra_img_local": normalized(rng, 80, 16),
                  "sim_qg": ((tg @ ig.T + tl @ il.T) / 2).astype(np.float32)}
        for name, array in arrays.items():
            np.save(path / f"{name}.npy", array)
        gpid = np.repeat(np.arange(20), 5)
        qpid = gpid[source]
        remap = path / "pid_remap_table.json"
        json_write(remap, {"evaluation_original_to_contiguous": {str(i): i for i in range(20)}})
        gallery_paths = [f"mock/{i}.jpg" for i in range(100)]
        if dataset == "tbapr":
            gallery_paths[1] = gallery_paths[0]  # duplicate within identity
            gallery_paths[10] = gallery_paths[0]  # conflicting PID path
        meta = {"dataset": dataset, "split": split, "epoch": 10, "seed": seed,
                "query_pid": qpid.tolist(), "gallery_pid": gpid.tolist(),
                "gallery_paths": gallery_paths, "pid_remap_table": str(remap),
                "query_source_gallery_idx": source.tolist(),
                "query_view": (["aerial"] * 200 if dataset == "aeri" else
                               ["img"] * 200 if dataset == "tbapr" else None),
                "gallery_view": (["aerial"] * 100 if dataset == "aeri" else
                                 ["img"] * 100 if dataset == "tbapr" else None),
                "query_source_subset": np.where(qpid < 10, "AG-ReID", "G2APS-AG").tolist(),
                "gallery_source_subset": np.where(gpid < 10, "AG-ReID", "G2APS-AG").tolist(),
                "metrics": {}}
        json_write(path / "meta.json", meta)
        json_write(path / "extra_meta.json", {"pid": np.tile(np.arange(20), 4).tolist(), "view": ["ground"] * 80})
        (path / "checkpoint_last.pth").touch()
        paths[(dataset, split)] = path
    cohorts = {name: {"cohort_A": list(range(10)), "nq": 100, "positive_union": 50, "pool": 50, "K": 10}
               for name in ("CUHK-PEDES", "ICFG-PEDES", "AERI-PEDES", "TBAPR-merged")}
    cohort = root / "cohorts.json"
    json_write(cohort, {"datasets": cohorts})
    tbapr = paths[("tbapr", "test")]
    meta = json.loads((tbapr / "meta.json").read_text())
    subsets = np.asarray(meta["query_source_subset"])
    texts = [f"{subsets[i]} a person wearing color{i % 5} row{i % 50}" for i in range(200)]
    dsroot = root / "TBAGPR"
    for name, filename in (("AG-ReID", "agtext_test"), ("G2APS-AG", "g2aps_test")):
        json_write(dsroot / name / f"{filename}.json", {"captions": np.asarray(texts)[subsets == name].tolist()})
    return paths, cohort, dsroot


class FixtureSubstitutions(ast.NodeTransformer):
    def __init__(self, assignments=None, reference=False):
        self.assignments = assignments or {}
        self.reference = reference

    def visit_Assign(self, node):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if len(names) == 1 and names[0] in self.assignments:
            node.value = ast.parse(repr(self.assignments[names[0]]), mode="eval").body
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Tuple)):
            for index, target in enumerate(node.targets[0].elts):
                if isinstance(target, ast.Name) and target.id in self.assignments:
                    node.value.elts[index] = ast.parse(repr(self.assignments[target.id]), mode="eval").body
        return self.generic_visit(node)

    def visit_Constant(self, node):
        # Documented MOCK cardinality/loop substitutions applied equally.
        if type(node.value) is int and node.value in (4220, 3200, 2000):
            return ast.copy_location(ast.Constant({4220: 80, 3200: 200, 2000: 24}[node.value]), node)
        if self.reference and node.value == "/root/autodl-tmp/_locks/cohort_splits_seed2026.json":
            return ast.copy_location(ast.Constant(self.assignments["cohort_path"]), node)
        if self.reference and node.value == "/root/autodl-tmp/_e_ext/s7a/tbapr_r_inductive.npy":
            return ast.copy_location(ast.Constant(str(Path(self.assignments["S7A"]) / "tbapr_r_inductive.npy")), node)
        return node


def execute(name, args, *, reference=False, overrides=None):
    path = (REFERENCE if reference else EXT) / f"{name}.py"
    module = types.ModuleType(f"scripts.ext._mock_{name}_{'old' if reference else 'new'}")
    module.__file__ = str(path)
    module.__package__ = "scripts.ext"
    tree = FixtureSubstitutions(overrides, reference).visit(ast.parse(path.read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    resource_stub = types.SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: types.SimpleNamespace(ru_maxrss=0))
    original_load = np.load
    def fixture_load(*positional, **kwargs):
        # Windows refuses cleanup of live mmap handles. The tiny fixture uses
        # RAM in both versions; array values/layout and math are unchanged.
        kwargs["mmap_mode"] = None
        return original_load(*positional, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(sys.modules, {"resource": resource_stub}), \
            warnings.catch_warnings(), mock.patch("numpy.load", side_effect=fixture_load):
        warnings.simplefilter("ignore", ResourceWarning)
        if reference:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
            with mock.patch("subprocess.check_output", return_value=head):
                exec(compile(tree, str(path), "exec"), module.__dict__)
            return module, None
        exec(compile(tree, str(path), "exec"), module.__dict__)
        for key, value in {"NBOOT": 24, "NULL": 3, "NULL_DRAWS": 3, "D_LEVELS": [10, 20, 30, 40]}.items():
            if key in module.__dict__:
                module.__dict__[key] = value
        return module, module.main(args)


def calculations(payload):
    if isinstance(payload, dict):
        return {key: calculations(value) for key, value in payload.items()
                if key not in ("provenance", "git_head", "argv", "peak_rss_gb", "elapsed_sec",
                               "run_dir", "seed", "artifacts", "paired_gallery_differences", "inputs_sha16")}
    if isinstance(payload, list):
        return [calculations(item) for item in payload]
    return payload


class RoundBExternalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="roundb_ext_mock_")
        cls.root = Path(cls.temp.name)
        cls.runs, cls.cohort, cls.dsroot = fixture(cls.root)
        cls.old = cls.root / "old"
        cls.old.mkdir()
        cls.new = cls.root / "new"
        cls.new.mkdir()
        cls.base_args = ["--root", str(cls.root / "runs"), "--tag", "mock_seed0", "--out", str(cls.new)]
        cls.ref_overrides = {"ROOT": str(cls.root / "runs"), "OUT": str(cls.old),
                             "S7A": str(cls.old), "DS": str(cls.dsroot), "cohort_path": str(cls.cohort),
                             "D_LEVELS": [10, 20, 30, 40]}
        # Original NBOOT/NULL are assigned as tuple targets: replace those
        # literal iteration budgets during compilation in BOTH variants.
        cls.original_visit_constant = FixtureSubstitutions.visit_Constant
        def budget(self, node):
            if type(node.value) is int and node.value == 10000:
                return ast.copy_location(ast.Constant(24), node)
            if type(node.value) is int and node.value == 300:
                return ast.copy_location(ast.Constant(3), node)
            return cls.original_visit_constant(self, node)
        FixtureSubstitutions.visit_Constant = budget
        cls.ref_a, _ = execute("S7a_csls", [], reference=True, overrides=cls.ref_overrides)
        cls.new_a, cls.result_a = execute("S7a_csls", cls.base_args)
        cls.s7a_json = next(cls.new.glob("S7a_csls_mock_seed0/*summary*.json"))

    @classmethod
    def tearDownClass(cls):
        FixtureSubstitutions.visit_Constant = cls.original_visit_constant
        cls.temp.cleanup()

    def assert_results_equal(self, old, new):
        # JSON serialisation also handles NaNs from the unchanged original
        # sensitivity diagnostics without declaring fabricated replacements.
        self.assertEqual(json.dumps(calculations(old), sort_keys=True),
                         json.dumps(calculations(new), sort_keys=True))

    def test_01_math_functions_verbatim(self):
        excluded = {"run_dirs", "latest", "main"}
        for name in NAMES:
            oldsrc = (REFERENCE / f"{name}.py").read_text(encoding="utf-8")
            baseline = subprocess.check_output(
                ["git", "show", f"8622dc2:scripts/ext/{name}.py"], cwd=REPO).decode("utf-8")
            self.assertEqual(oldsrc, baseline, f"{name} reference differs from baseline source")
            newsrc = (EXT / f"{name}.py").read_text(encoding="utf-8")
            oldf = {node.name: ast.get_source_segment(oldsrc, node) for node in ast.parse(oldsrc).body if isinstance(node, ast.FunctionDef)}
            newf = {node.name: ast.get_source_segment(newsrc, node) for node in ast.parse(newsrc).body if isinstance(node, ast.FunctionDef)}
            for func, source in oldf.items():
                if func in excluded:
                    continue
                current = newf[func]
                if name == "S7d_aeri" and func == "protocol":
                    current = current.replace("    PROTOCOL_CORRECT[name] = f0 < 1  # D3 addition: paired gallery-composition comparisons.\n", "")
                    current = current.replace('    PROTOCOL_CORRECTIONS[name] = {"transductive": f1 < 1, "inductive": f2 < 1}\n', "")
                self.assertEqual(source, current, f"{name}.{func} math/source changed")

    def test_02_s7a_reference_all_datasets(self):
        self.assert_results_equal(self.ref_a.summary, self.result_a)
        for ds in HELPERS.DATASETS:
            for mode in ("transductive", "inductive"):
                old = np.load(self.old / f"{ds}_r_{mode}.npy")
                new = np.load(self.result_a["datasets"][ds]["artifacts"]["r_" + mode])
                self.assertTrue(np.array_equal(old, new))
        print("MOCK S7a: 200x100, D_global=8 D_local=16; reference JSON + penalties array_equal=True")

    def test_03_s7b_reference(self):
        # The mock distractor pool has only 50 images; small levels preserve
        # the original grid algorithm while avoiding a degenerate full-only fit.
        old, _ = execute("S7b_distractor", [], reference=True, overrides=self.ref_overrides)
        _, new = execute("S7b_distractor", self.base_args + ["--cohort-json", str(self.cohort), "--s7a-json", str(self.s7a_json)])
        self.assert_results_equal(old.R, new)
        self.assertEqual(len(new["aeri"]["rows"]), 25)
        print("MOCK S7b: four datasets, five distractor levels/five draws; reference calculations exactly equal")

    def test_04_s7c_reference(self):
        old, _ = execute("S7c_valselect", [], reference=True, overrides=self.ref_overrides)
        _, new = execute("S7c_valselect", self.base_args)
        self.assert_results_equal(old.R, new)
        print("MOCK S7c: CUHK val selection + four posthoc test grids, all reference calculations exactly equal")

    def test_05_s7d_reference_and_paired(self):
        old, _ = execute("S7d_aeri", [], reference=True, overrides=self.ref_overrides)
        args = self.base_args + ["--run-dir", str(self.runs[("aeri", "test")])]
        module, new = execute("S7d_aeri", args)
        self.assert_results_equal(old.R, new)
        paired = new["paired_gallery_differences"]
        self.assertEqual(set(paired), {"ground_minus_official_aerial", "mixed_minus_official_aerial", "size_controlled_aerial_minus_official_aerial"})
        self.assertEqual(len(paired["size_controlled_aerial_minus_official_aerial"]["runs"]), 5)
        names = list(module.PROTOCOL_CORRECT)
        official = module.PROTOCOL_CORRECT[names[0]]
        target = module.PROTOCOL_CORRECT[names[1]]
        for mode in ("transductive", "inductive"):
            old_gain = module.PROTOCOL_CORRECTIONS[names[0]][mode].astype(float) - official
            new_gain = module.PROTOCOL_CORRECTIONS[names[1]][mode].astype(float) - target
            values = paired["ground_minus_official_aerial"]["correction_gain_difference"][mode]
            self.assertEqual(values["delta_gain_r1_pp"], float(100 * (new_gain.mean() - old_gain.mean())))
            self.assertEqual(values["identity_ci95"], module.boot(old_gain, new_gain))
        print("MOCK S7d: official/ground/mixed/five controlled draws; original fields exactly equal; paired identity intervals produced")

    def test_06_s7e_reference(self):
        old, _ = execute("S7e_tbapr", [], reference=True, overrides=self.ref_overrides)
        _, new = execute("S7e_tbapr", self.base_args + ["--run-dir", str(self.runs[("tbapr", "test")]), "--dataset-root", str(self.dsroot), "--s7a-json", str(self.s7a_json)])
        self.assert_results_equal(old.R, new)
        self.assertEqual(len(new["variants"]), 8)
        self.assertTrue(all(row["nq"] > 0 and row["ng"] > 0 for row in new["variants"].values()))
        print("MOCK S7e: eight row/text sensitivity variants; matching S7a manifest; all reference calculations exactly equal")

    def test_07_help_no_io(self):
        for name in NAMES + ("S7_aggregate_seeds",):
            result = subprocess.run([sys.executable, str(EXT / f"{name}.py"), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--tag", result.stdout)
            self.assertIn("--out", result.stdout)

    def test_08_ambiguous_runs_rejected(self):
        args = HELPERS.parse_common(HELPERS.common_parser("mock", "mock"), self.base_args)
        with tempfile.TemporaryDirectory() as tmp:
            duplicate = Path(tmp) / "runs"
            duplicate.mkdir()
            for number in (0, 1):
                path = duplicate / f"aeri_{number}"
                path.mkdir()
                meta = json.loads((self.runs[("aeri", "test")] / "meta.json").read_text())
                json_write(path / "meta.json", meta)
                np.save(path / "sim_qg.npy", np.zeros((200, 100), np.float32))
            args.root = str(duplicate)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                HELPERS.select_runs(args)

    def test_09_no_overwrite_and_provenance(self):
        args = HELPERS.parse_common(HELPERS.common_parser("exclusive", "mock"), self.base_args)
        first, second = HELPERS.result_directory(args), HELPERS.result_directory(args)
        self.assertNotEqual(first, second)
        path = Path(first) / "one_mock.npy"
        HELPERS.save_npy(path, np.ones(2))
        with self.assertRaises(FileExistsError):
            HELPERS.save_npy(path, np.zeros(2))
        self.assertTrue(np.array_equal(np.load(path), np.ones(2)))
        evidence = self.result_a["provenance"]
        self.assertEqual(len(evidence["git_HEAD"]), 40)
        self.assertTrue(evidence["argv"])
        for item in evidence["inputs"]:
            self.assertEqual(len(item["sha256_16"]), 16)
            self.assertEqual(item["bytes"], Path(item["path"]).stat().st_size)

    def test_10_s7a_binding_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            HELPERS.load_s7a_artifact(self.s7a_json, self.runs[("cuhkpedes", "val")], "cuhkpedes", "inductive")

    def test_11_aggregate_three_actual_mock_outputs(self):
        paths = [self.s7a_json]
        for seed in (1, 2):
            root = self.root / f"seed{seed}"
            fixture(root, seed)
            _, result = execute("S7a_csls", ["--root", str(root / "runs"), "--out", str(self.new), "--tag", f"mock_seed{seed}"])
            paths.append(next(self.new.glob(f"S7a_csls_mock_seed{seed}/*summary*.json")))
        args = ["--out", str(self.new), "--tag", "mock_three_seeds"]
        for path in paths:
            args.extend(["--s7a-json", str(path)])
        with contextlib.redirect_stdout(io.StringIO()):
            output = HELPERS.main(args)
        for dataset, row in output["datasets"].items():
            self.assertEqual([item["seed"] for item in row["per_seed"]], [0, 1, 2])
            for mode in ("transductive", "inductive"):
                counts = row["summary"][mode]["sign_counts"]
                self.assertEqual(counts["positive"] + counts["zero"] + counts["negative"], 3)
            self.assertNotIn("significant", json.dumps(row))
        print("MOCK aggregate: three independently executed S7a mock seed outputs; four datasets; count conservation passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
