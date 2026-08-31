"""S15 200x100 synthetic tests; no real checkpoint, annotation or metric claims."""

import builtins
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None
import numpy as np

from . import S15_proxy_validation as s15
from ._offline_common import annotation_from_records, compact_metrics, evaluate, topk_mean


def fixture():
    rng = np.random.default_rng(2026)

    def features(count, prefix):
        output = {}
        for key, width in (("feat", 4), ("tse_feat", 6)):
            values = rng.normal(size=(count, width)).astype(np.float32)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            output[prefix + "_" + key] = values
        return output

    train = [{"file_path": f"G2APS/{'D' if i % 2 == 0 else 'S'}2_{i:04d}.jpg", "id": 1000 + i // 2,
              "cam_id": i % 2, "split": "train", "captions": [f"train caption {i} a", f"train caption {i} b"]}
             for i in range(200)]
    test = [{"file_path": f"G2APS/{'D' if i % 2 == 0 else 'S'}1_{i:04d}.jpg", "id": i // 2,
             "cam_id": i % 2, "split": "test", "captions": [f"test caption {i} a", f"test caption {i} b"]}
            for i in range(100)]
    image_train, text_train = features(200, "image"), features(400, "text")
    image_test, text_test = features(100, "image"), features(200, "text")

    def scores(text, gallery):
        return (text["text_feat"] @ gallery["image_feat"].T +
                text["text_tse_feat"] @ gallery["image_tse_feat"].T) / 2

    return SimpleNamespace(train=train, test=test, image_train=image_train, text_train=text_train,
                           image_test=image_test, text_test=text_test, ann=annotation_from_records(test),
                           qg=scores(text_test, image_test), bank=scores(text_train, image_test), scores=scores)


class PureProxyTests(unittest.TestCase):
    def test_partition_200x100_conservation_and_replay(self):
        f = fixture()
        vr, br, ann, split = s15.split_training_records(f.train, 50, 2026)
        expected = np.random.default_rng(2026).permutation(np.unique([r["id"] for r in f.train]))
        self.assertEqual(split["validation_identities"], expected[:50].tolist())
        self.assertEqual(split["bank_identities"], expected[50:].tolist())
        self.assertTrue(all(split["checks"].values()))
        self.assertEqual((len(ann.query_ids), len(ann.gallery_ids)), (200, 100))
        self.assertEqual([r["file_path"] for r in vr], [r["file_path"] for r in f.train if r["id"] in expected[:50]])
        self.assertEqual(len(vr) + len(br), 200)
        print("S15 MOCK PARTITION " + json.dumps(split["counts"], sort_keys=True))

    def test_proxy_excludes_source_and_reports_no_positive(self):
        f = fixture()
        # A source-perfect diagonal must NOT produce source-inclusive R1=100.
        scores = np.zeros((200, 100), np.float32)
        scores[np.arange(200), f.ann.query_source_indices] = 10
        raw = evaluate(scores, f.ann, exclude_source=False)
        result = s15.scan_proxy(scores, np.zeros((200, 100), np.float32), f.ann)
        self.assertEqual(raw["R1"], 100)
        self.assertLess(result["uncorrected"]["R1"], 100)
        self.assertEqual(len(result["rows"]), 20)
        self.assertEqual((result["selected"]["alpha"], result["selected"]["k"]), (.25, 5))
        self.assertEqual((result["runner_up"]["alpha"], result["runner_up"]["k"]), (.5, 5))
        singleton = [dict(row, id=10_000 + i if i == 0 else row["id"]) for i, row in enumerate(f.test)]
        ann = annotation_from_records(singleton)
        result = s15.scan_proxy(scores, np.zeros((200, 100), np.float32), ann)
        # Both the singleton and its now-singleton former pair have no positive.
        self.assertEqual(result["n_query_skipped_no_positive"], 4)
        self.assertEqual(result["uncorrected"]["n_query"], 196)

    def test_frozen_single_pass_matches_exact_shared_evaluator_and_deduplicates(self):
        f = fixture()
        for selected in ({"alpha": .75, "k": 20}, {"alpha": .5, "k": 10}):
            result = s15.evaluate_frozen_methods(f.qg, f.bank, f.ann, selected, block_size=17)
            raw = evaluate(f.qg, f.ann)
            chosen = evaluate(f.qg, f.ann, penalty=topk_mean(f.bank, selected["k"]), alpha=selected["alpha"])
            default = evaluate(f.qg, f.ann, penalty=topk_mean(f.bank, 10), alpha=.5)
            for name, ref in (("uncorrected", raw), ("selected", chosen), ("literature_default", default)):
                for key in s15.METRICS:
                    self.assertEqual(result[name][key], ref[key])
            self.assertEqual(result["test_evaluation_passes"], 1)
            self.assertEqual(result["rows_visited"], 200)
            self.assertEqual(result["unique_corrected_settings"], 1 if selected == {"alpha": .5, "k": 10} else 2)
        print("S15 MOCK FROZEN METRICS exact_error_pp=0 shape=[200,100] traversal=1 default_dedup=PASS")

    def test_invalid_inputs(self):
        f = fixture()
        for count in (0, 100, 101):
            with self.assertRaises(ValueError):
                s15.split_training_records(f.train, count)
        with self.assertRaisesRegex(ValueError, "non-training"):
            s15.split_training_records(f.test, 20)
        with self.assertRaisesRegex(ValueError, "at least"):
            s15.scan_proxy(f.qg, f.bank[:29], f.ann)
        with self.assertRaisesRegex(ValueError, "Locked parameters"):
            s15.evaluate_frozen_methods(f.qg, f.bank, f.ann, {"alpha": .123, "k": 10})
        for command in (("--help",), ("--seed", "-1")):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                s15.main(list(command))


@unittest.skipIf(torch is None, "PyTorch needed for real CPU scoring and RNG/buffer guards")
class ProxyPipelineTests(unittest.TestCase):
    def pipeline(self, corrupt_rng=False, corrupt_buffer=False, wrong_raw=False,
                 alter_test=False, missing_e3=False, repeat=False, missing_image=False):
        f = fixture()
        with tempfile.TemporaryDirectory(prefix="tagpr-s15-mock-") as temp:
            root, events = Path(temp), []
            out = root / "_e1_s15"
            out.mkdir()
            images = root / "images"
            (images / "G2APS").mkdir(parents=True)
            for row in f.train:
                (images / row["file_path"]).write_bytes(b"mock image; extraction is synthetic")
            train_path, test_path = root / "train_reid.json", root / "test_reid.json"
            train_path.write_text(json.dumps(f.train), encoding="utf-8")
            test_path.write_text(json.dumps(f.test), encoding="utf-8")
            config_path, checkpoint = root / "config.yaml", root / "checkpoint.pth"
            config_path.write_text("# mock configuration\n", encoding="utf-8")
            checkpoint.write_bytes(b"mock checkpoint, never loaded")
            qg_path, bank_path = root / "E3_sim_qg_fresh.npy", root / "E3_sim_bank_g_fresh.npy"
            if alter_test:
                f.qg = -f.qg
            np.save(qg_path, f.qg)
            np.save(bank_path, f.bank)
            provenance = {"checkpoint": str(checkpoint), "config": str(config_path), "backbone": "ViT-L/14",
                          "loaded_state": "model", "seed": 0}
            authority = {"status": "complete", "fresh_forward": True,
                         "pure_query_independent_inductive_pipeline": True, "alignment_W_used": False,
                         "feature_provenance": provenance, "image_batch_size": 32,
                         "train_annotation": {"identities": 100, "captions": 400},
                         "test_annotation": {"path": str(test_path)}, "annotation": f.ann.summary,
                         "text_fingerprints": {"train": s15.e3.text_fingerprint(
                             [text for row in f.train for text in row["captions"]])},
                         "fresh_files": {"sim_qg": str(qg_path), "sim_bank_g": str(bank_path),
                                         "sim_gg": "unused_gg_file_must_not_be_opened.npy"},
                         "fresh_shapes": {"sim_qg": [200, 100], "sim_bank_g": [400, 100]},
                         "fresh_metrics": compact_metrics(evaluate(f.qg, f.ann))}
            if wrong_raw:
                authority["fresh_metrics"]["R1"] += 1.0
            e3_path = root / "E3_mock.json"
            if not missing_e3:
                e3_path.write_text(json.dumps(authority), encoding="utf-8")
            protected = {e3_path.resolve(), test_path.resolve(), qg_path.resolve(), bank_path.resolve()}
            input_paths = [train_path, test_path, checkpoint, config_path, qg_path, bank_path]
            if not missing_e3:
                input_paths.append(e3_path)
            input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
            config = SimpleNamespace(experiment=SimpleNamespace(max_words=30, text_length=77, view_prompt=False),
                                     image_dir=str(images))
            by_path = {row["file_path"]: index for index, row in enumerate(f.train)}
            by_caption = {caption: 2 * i + j for i, row in enumerate(f.train) for j, caption in enumerate(row["captions"])}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("statistics", torch.zeros(2))

            def load(*pos, **kwargs):
                self.assertEqual(pos[4], 100, "checkpoint class count must use full train IDs")
                self.assertEqual(pos[5], 0)
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                events.append("model")
                return Model(), config, provenance

            def gallery(model, cfg, ann_path, image_root, device, image_batch_size, num_workers):
                self.assertEqual(num_workers, 2)
                self.assertTrue(str(ann_path).endswith("proxy_validation_reid.json"))
                rows = json.loads(Path(ann_path).read_text(encoding="utf-8"))
                ids = [by_path[row["file_path"]] for row in rows]
                self.assertEqual(len(ids), 100)
                events.append("gallery_V")
                torch.rand(1)  # One stochastic image pass, never reset/retried.
                return {key: value[ids] for key, value in f.image_train.items()}

            def text(model, texts, **kwargs):
                events.append("text_V" if "text_V" not in events else "text_B")
                self.assertEqual(len(texts), 200)
                indices = [by_caption[value] for value in texts]
                if corrupt_rng:
                    torch.rand(1)
                if corrupt_buffer:
                    model.statistics.add_(1)
                return {key: value[indices] for key, value in f.text_train.items()}

            original_path_open, original_open = Path.open, builtins.open

            def check(path):
                if isinstance(path, (str, Path)) and Path(path).resolve() in protected:
                    if not list(out.glob("parameters_lock*.json")):
                        raise AssertionError("TEST INPUT OPENED BEFORE LOCK: " + str(path))
                    events.append("test_read:" + Path(path).name)

            def checked_path_open(path, *args, **kwargs):
                check(path)
                return original_path_open(path, *args, **kwargs)

            def checked_open(path, *args, **kwargs):
                check(path)
                return original_open(path, *args, **kwargs)

            args = ["--train-anno", str(train_path), "--config", str(config_path), "--backbone", "ViT-L/14",
                    "--resume", str(checkpoint), "--out", str(out), "--device", "cpu",
                    "--e3-json", str(e3_path), "--val-identities", "50", "--score-block-size", "17"]
            if missing_image:
                args += ["--image-root", str(root / "missing_images")]
            capture = io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(patch.object(s15.e3, "load_model", side_effect=load))
                stack.enter_context(patch.object(s15.e3, "extract_gallery_features", side_effect=gallery))
                stack.enter_context(patch.object(s15.e3, "extract_text_features", side_effect=text))
                stack.enter_context(patch.object(s15.e3, "preprocess_texts", side_effect=lambda records, _: [
                    text for row in records for text in row["captions"]]))
                stack.enter_context(patch.object(Path, "open", checked_path_open))
                stack.enter_context(patch.object(builtins, "open", checked_open))
                spy = stack.enter_context(patch.object(s15, "final_test_phase", wraps=s15.final_test_phase))
                with redirect_stdout(capture):
                    status = s15.main(args)
                    previous = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
                    if repeat:
                        self.assertEqual(s15.main(args), status)
                        for name, digest in previous.items():
                            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest)
                final_calls = spy.call_count
            reports = []
            for path in out.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("task") == "S15":
                    reports.append(payload)
            self.assertTrue(reports, "failure/success must both write final JSON")
            self.assertTrue(list(out.glob("*.log")))
            self.assertEqual(input_hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths})
            report = sorted(reports, key=lambda r: r["started_utc"])[-1]
            self.assertTrue(report["argv"])
            self.assertTrue(all(len(i["sha256_16"]) == 16 for i in report["inputs"].values()))
            return status, report, events, final_calls, capture.getvalue()

    def test_complete_real_orchestration_200x100_no_test_reads_until_lock(self):
        status, report, events, calls, _ = self.pipeline()
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["proxy_shapes"]["sim_qg"], [200, 100])
        self.assertEqual(len(report["proxy"]["rows"]), 20)
        self.assertEqual(calls, 1)
        self.assertEqual(events[:4], ["model", "gallery_V", "text_V", "text_B"])
        self.assertEqual(events.count("gallery_V"), 1)
        self.assertEqual(report["test"]["test_evaluation_passes"], 1)
        self.assertLessEqual(report["test"]["unique_corrected_settings"], 2)
        self.assertTrue(all(guard["rng_unchanged"] for guard in report["independence_guards"]))
        self.assertFalse(report["test_matrices_recomputed"])
        print("S15 MOCK PIPELINE " + json.dumps({"status": report["status"], "proxy_shape": [200, 100],
            "grid_rows": len(report["proxy"]["rows"]), "selected": report["test"]["frozen_parameters"]["selected"],
            "model_num_classes": report["model_num_classes"], "final_phase_calls": calls,
            "gallery_passes": report["gallery_extraction_passes"], "all_input_files_unchanged": True}, sort_keys=True))

    def test_test_scores_cannot_change_proxy_selection(self):
        a, b = self.pipeline(), self.pipeline(alter_test=True)
        self.assertEqual(a[0], 0)
        self.assertEqual(b[0], 0)
        self.assertEqual(a[1]["proxy"], b[1]["proxy"])
        self.assertNotEqual(a[1]["test"]["uncorrected"], b[1]["test"]["uncorrected"])

    def test_rng_and_buffer_changes_stop_before_test(self):
        for option in ("corrupt_rng", "corrupt_buffer"):
            status, report, events, calls, log = self.pipeline(**{option: True})
            self.assertEqual(status, 1)
            self.assertEqual(calls, 0)
            self.assertFalse(any(e.startswith("test_read:") for e in events))
            self.assertNotIn("parameter_lock", report)
            self.assertIn("independence guard", log)

    def test_raw_e3_mismatch_stops_without_retry_or_recompute(self):
        status, report, events, calls, log = self.pipeline(wrong_raw=True)
        self.assertEqual(status, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(events.count("gallery_V"), 1)
        self.assertIn("E3 uncorrected exact R1", log)
        self.assertIn("parameter_lock", report)
        self.assertNotIn("test", report)

    def test_missing_e3_is_only_discovered_after_lock(self):
        status, report, events, calls, _ = self.pipeline(missing_e3=True)
        self.assertEqual(status, 1)
        self.assertEqual(calls, 1)
        self.assertIn("parameter_lock", report)
        self.assertEqual(events[:4], ["model", "gallery_V", "text_V", "text_B"])

    def test_missing_proxy_image_fails_and_writes_report(self):
        status, report, events, calls, _ = self.pipeline(missing_image=True)
        self.assertEqual(status, 1)
        self.assertEqual(calls, 0)
        self.assertNotIn("gallery_V", events)

    def test_repeat_never_overwrites_any_existing_output(self):
        status, report, events, calls, _ = self.pipeline(repeat=True)
        self.assertEqual(status, 0)
        self.assertEqual(calls, 2)
        self.assertEqual(events.count("gallery_V"), 2)


if __name__ == "__main__":
    unittest.main()
