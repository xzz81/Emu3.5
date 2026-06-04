import math
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from src.utils.entropy_trace import (
    build_trace_records,
    calibration_bin_rows,
    component_correlation_rows,
    hallucination_diagnostic_rows,
    normalized_entropy_from_scores,
    read_jsonl,
    render_summary_report,
    summarize_records,
    thinking_transition_rows,
    write_html_report,
    write_jsonl,
    write_visual_artifacts,
)
from src.utils.activation_probe import ActivationAccumulator, token_group_labels, token_type_for_id


SPECIAL_IDS = {
    "BOS": 0,
    "EOS": 1,
    "BOG": 2,
    "EOG": 3,
    "BOC": 4,
    "EOC": 5,
    "BOI": 6,
    "IMG": 7,
    "EOI": 8,
    "EOL": 9,
    "BOV": 20,
}


class EntropyTraceTest(unittest.TestCase):
    def test_normalized_entropy_extremes(self):
        uniform = torch.zeros(4)
        self.assertAlmostEqual(normalized_entropy_from_scores(uniform), 1.0, places=6)

        sharp = torch.tensor([0.0, -math.inf, -math.inf, -math.inf])
        self.assertEqual(normalized_entropy_from_scores(sharp), 0.0)

    def test_segments_and_token_types(self):
        generated = [2, 11, 3, 6, 7, 20, 9, 21, 8, 1]
        scores = [torch.zeros(23) for _ in generated]
        records = build_trace_records(
            scores,
            generated,
            sample_id="synthetic",
            special_token_ids=SPECIAL_IDS,
            visual_token_start=20,
        )

        self.assertEqual(records[0]["token_type"], "structure")
        self.assertEqual(records[1]["segment"], "global_cot")
        self.assertEqual(records[1]["token_type"], "thinking")
        self.assertEqual(records[5]["segment"], "visual")
        self.assertEqual(records[5]["token_type"], "visual")
        self.assertEqual(records[6]["token_type"], "structure")
        self.assertEqual(records[-1]["segment"], "text")

    def test_activation_probe_grouping_and_contrast(self):
        input_ids = torch.tensor([[0, 11, 20, 8]])
        labels = token_group_labels(input_ids, "generation", SPECIAL_IDS, visual_token_start=20)
        self.assertEqual(token_type_for_id(20, SPECIAL_IDS, visual_token_start=20), "visual")
        self.assertIn("task=generation", labels[0])
        self.assertIn("token_type=visual", labels[2])

        acc = ActivationAccumulator()
        gen = torch.tensor([[[0.0, 3.0], [0.0, 4.0]]])
        read = torch.tensor([[[2.0, 0.0], [3.0, 0.0]]])
        acc.update(0, [("task=generation",), ("task=generation",)], gen)
        acc.update(0, [("task=understanding",), ("task=understanding",)], read)
        rows = acc.top_contrast_rows("task=generation", "task=understanding", top_k=1)
        self.assertEqual(rows[0]["layer"], 0)
        self.assertEqual(rows[0]["neuron"], 1)
        self.assertGreater(rows[0]["delta_mean_abs"], 0.0)

    def test_cfg_trace_contributes_to_ume(self):
        generated = [11]
        scores = [torch.zeros(13)]
        without_cfg = build_trace_records(
            scores,
            generated,
            sample_id="a",
            special_token_ids=SPECIAL_IDS,
            visual_token_start=20,
        )[0]
        with_cfg = build_trace_records(
            scores,
            generated,
            sample_id="a",
            special_token_ids=SPECIAL_IDS,
            cfg_trace=[{"u_cfg": 0.75}],
            visual_token_start=20,
        )[0]
        self.assertFalse(without_cfg["u_cfg_available"])
        self.assertTrue(with_cfg["u_cfg_available"])
        self.assertGreater(with_cfg["ume"], without_cfg["ume"])

    def test_visual_full_and_sample_entropy_versions(self):
        generated = [20]
        scores = [torch.tensor([0.0, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf])]
        record = build_trace_records(
            scores,
            generated,
            sample_id="visual",
            special_token_ids=SPECIAL_IDS,
            cfg_trace=[{"u_visual_full": 0.75, "visual_full_candidate_count": 131072}],
            visual_token_start=20,
        )[0]

        self.assertEqual(record["token_type"], "visual")
        self.assertEqual(record["candidate_count_sample"], 1)
        self.assertEqual(record["candidate_count_full"], 131072)
        self.assertEqual(record["u_tok_sample"], record["u_tok"])
        self.assertAlmostEqual(record["u_visual_full"], 0.75)
        self.assertGreater(record["ume_full"], record["ume_sample"])

    def test_jsonl_and_summary_report(self):
        scores = [torch.zeros(23), torch.zeros(23)]
        records = build_trace_records(
            scores,
            [11, 12],
            sample_id="io",
            special_token_ids=SPECIAL_IDS,
            visual_token_start=20,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            write_jsonl(records, path)
            loaded = read_jsonl(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(summarize_records(loaded, "token_type")[0]["count"], 2)
        self.assertEqual(len(component_correlation_rows(loaded)), 5)
        self.assertIn("Highest UME tokens", render_summary_report(loaded, "Synthetic"))

    def test_metadata_task_and_thinking_transition(self):
        generated = [2, 11, 12, 3, 6, 13, 7, 20, 21, 8, 1]
        scores = [torch.zeros(23) for _ in generated]
        records = build_trace_records(
            scores,
            generated,
            sample_id="story",
            special_token_ids=SPECIAL_IDS,
            cfg_trace=[{"u_cfg": 0.0} for _ in generated],
            visual_token_start=20,
            metadata={"task": "story"},
        )
        self.assertEqual(summarize_records(records, "task")[0]["task"], "story")
        rows = thinking_transition_rows(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "story")
        self.assertEqual(rows[0]["task"], "story")

    def test_visual_artifacts_are_written(self):
        generated = [11, 12, 6, 13, 7, 20, 21, 8, 1]
        scores = [torch.zeros(23) for _ in generated]
        records = build_trace_records(
            scores,
            generated,
            sample_id="viz",
            special_token_ids=SPECIAL_IDS,
            visual_token_start=20,
            metadata={"task": "t2i"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            figures = write_visual_artifacts(records, out_dir, max_traces=2)
            html_path = write_html_report(records, out_dir, "Viz", figures)
            self.assertTrue((out_dir / "figures" / "ume_histogram.svg").exists())
            self.assertTrue((out_dir / "figures" / "ume_by_token_type_boxplot.svg").exists())
            self.assertTrue(html_path.exists())
            self.assertIn("UME by task", html_path.read_text(encoding="utf-8"))

    def test_hallucination_diagnostics_and_calibration(self):
        records = [
            {"task": "t2i", "ume": 0.10, "u_cfg": 0.70, "is_error": True},
            {"task": "t2i", "ume": 0.85, "u_cfg": 0.10, "is_error": True},
            {"task": "t2i", "ume": 0.30, "u_cfg": 0.05, "is_error": False},
            {"task": "t2i", "ume": 0.60, "u_cfg": 0.05, "is_error": False},
        ]
        rows = hallucination_diagnostic_rows(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error_tokens"], 2)
        self.assertEqual(rows[0]["false_conf_tokens"], 1)
        self.assertEqual(rows[0]["unresolved_high_ume_tokens"], 1)
        self.assertEqual(rows[0]["low_ume_high_cfg_tokens"], 1)
        self.assertGreaterEqual(rows[0]["auc_ume_error"], 0.0)

        calibration = calibration_bin_rows(records, bins=5)
        self.assertGreaterEqual(len(calibration), 3)
        self.assertIn("running_ece", calibration[-1])

    def test_real_run_summary_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_real_ume_runs.py"
        spec = importlib.util.spec_from_file_location("summarize_real_ume_runs", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        records = [
            {
                "task": "t2i",
                "sample_id": "case",
                "step": 0,
                "token_text": "<|image start|>",
                "token_type": "structure",
                "segment": "text",
                "ume": 0.0,
                "u_cfg": 0.0,
            },
            {
                "task": "t2i",
                "sample_id": "case",
                "step": 1,
                "token_text": "<|visual token 000001|>",
                "token_type": "visual",
                "segment": "visual",
                "ume": 0.4,
                "u_cfg": 0.2,
            },
            {
                "task": "t2i",
                "sample_id": "case",
                "step": 2,
                "token_text": "<|image end|>",
                "token_type": "structure",
                "segment": "visual",
                "ume": 0.0,
                "u_cfg": 0.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs" / "emu3p5-image"
            run_dir = root / "t2i" / "ume_trace_runs" / "run"
            trace_dir = run_dir / "entropy_traces"
            raw_dir = run_dir / "raw_generations"
            decoded_dir = run_dir / "decoded"
            trace_dir.mkdir(parents=True)
            raw_dir.mkdir()
            decoded_dir.mkdir()
            write_jsonl(records, trace_dir / "case_entropy.jsonl")
            (raw_dir / "case.txt").write_text("<|image start|><|image end|>", encoding="utf-8")
            Image.new("RGB", (4, 3)).save(decoded_dir / "case_image_00.png")

            sample = module.sample_row(run_dir, trace_dir / "case_entropy.jsonl", root)
            self.assertEqual(sample["task"], "t2i")
            self.assertTrue(sample["decoded"])
            self.assertTrue(sample["image_complete"])
            self.assertEqual(sample["visual_tokens"], 1)
            self.assertEqual(sample["decoded_width"], 4)
            self.assertAlmostEqual(sample["visual_mean_ume"], 0.4)

            run = module.run_row(run_dir, [sample])
            self.assertEqual(run["samples"], 1)
            self.assertEqual(run["image_complete_samples"], 1)

    def test_manual_sample_label_analysis_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_manual_sample_labels.py"
        spec = importlib.util.spec_from_file_location("analyze_manual_sample_labels", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        sample_rows = [
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "ok",
                "decoded": "True",
                "image_complete": "True",
                "eos": "False",
                "visual_tokens": "2",
                "visual_mean_ume": "0.1",
                "visual_p90_ume": "0.2",
                "visual_max_ume": "0.3",
                "visual_mean_u_cfg": "0.01",
                "visual_max_u_cfg": "0.02",
            },
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "bad",
                "decoded": "True",
                "image_complete": "True",
                "eos": "False",
                "visual_tokens": "2",
                "visual_mean_ume": "0.8",
                "visual_p90_ume": "0.9",
                "visual_max_ume": "1.0",
                "visual_mean_u_cfg": "0.03",
                "visual_max_u_cfg": "0.04",
            },
        ]
        labels = [
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "ok",
                "verdict": "success",
                "quality_score": 1.0,
                "is_error": False,
                "include_in_analysis": True,
                "reason": "ok",
            },
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "bad",
                "verdict": "failure",
                "quality_score": 0.0,
                "is_error": True,
                "include_in_analysis": True,
                "reason": "bad",
            },
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "missing",
                "verdict": "unverifiable",
                "quality_score": "",
                "is_error": "",
                "include_in_analysis": False,
                "reason": "not in sample rows",
            },
        ]
        joined = module.join_rows(sample_rows, labels)
        self.assertEqual(len(joined), 3)
        self.assertEqual(sum(1 for row in joined if row["matched"]), 2)
        analyzed = module.analyzable(joined)
        self.assertEqual(len(analyzed), 2)
        summaries = module.summary_rows(joined)
        by_group = {row["group"]: row for row in summaries}
        self.assertEqual(by_group["all"]["samples"], 2)
        self.assertEqual(by_group["all"]["errors"], 1)
        self.assertEqual(by_group["all"]["auc_visual_mean_ume_error"], 1.0)

    def test_apply_region_annotations(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "apply_region_annotations.py"
        spec = importlib.util.spec_from_file_location("apply_region_annotations", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        records = [
            {"sample_id": "case", "token_type": "structure", "segment": "text", "token_text": "<|image start|>", "ume": 0.0},
            {"sample_id": "case", "token_type": "visual", "segment": "visual", "token_text": "<|visual token 000001|>", "ume": 0.2},
            {"sample_id": "case", "token_type": "visual", "segment": "visual", "token_text": "<|visual token 000002|>", "ume": 0.3},
            {"sample_id": "case", "token_type": "structure", "segment": "visual", "token_text": "<|extra_200|>", "ume": 0.0},
            {"sample_id": "case", "token_type": "visual", "segment": "visual", "token_text": "<|visual token 000003|>", "ume": 0.4},
            {"sample_id": "case", "token_type": "visual", "segment": "visual", "token_text": "<|visual token 000004|>", "ume": 0.5},
        ]
        annotation = {
            "sample_id": "case",
            "token_grid": {"height": 2, "width": 2},
            "failure_type": "region_error",
            "error_region_token_bbox": {"row_min": 1, "row_max": 1, "col_min": 0, "col_max": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            trace_dir = run_dir / "entropy_traces"
            ann_dir = run_dir / "manual_annotations"
            out_dir = run_dir / "entropy_traces_labeled_regions"
            trace_dir.mkdir(parents=True)
            ann_dir.mkdir()
            write_jsonl(records, trace_dir / "case_entropy.jsonl")
            (ann_dir / "case_manual_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")

            count = module.apply_annotation(run_dir, ann_dir / "case_manual_annotation.json", out_dir)
            self.assertEqual(count, len(records))
            labeled = read_jsonl(out_dir / "case_entropy.jsonl")
            visual = [row for row in labeled if row["token_type"] == "visual"]
            self.assertEqual([(row["visual_row"], row["visual_col"]) for row in visual], [(0, 0), (0, 1), (1, 0), (1, 1)])
            self.assertFalse(visual[0]["is_error"])
            self.assertTrue(visual[2]["is_error"])
            self.assertEqual(visual[2]["error_type"], "region_error")

    def test_ume_weight_sweep_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ume_weight_sweep.py"
        spec = importlib.util.spec_from_file_location("analyze_ume_weight_sweep", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        specs = module.metric_specs(0.5)
        self.assertTrue(any(spec["metric"] == "ume_default" for spec in specs))
        rows = [
            {"task": "t2i", "is_error": False, "ume": 0.8, "u_tok": 0.8, "u_intra": 0.8, "u_cfg": 0.0, "u_mod": 0.0},
            {"task": "t2i", "is_error": True, "ume": 0.1, "u_tok": 0.1, "u_intra": 0.1, "u_cfg": 0.0, "u_mod": 0.0},
            {"task": "t2i", "is_error": False, "ume": 0.7, "u_tok": 0.7, "u_intra": 0.7, "u_cfg": 0.0, "u_mod": 0.0},
            {"task": "t2i", "is_error": True, "ume": 0.2, "u_tok": 0.2, "u_intra": 0.2, "u_cfg": 0.0, "u_mod": 0.0},
        ]
        summary = module.summarize_scores(rows, [spec for spec in specs if spec["metric"] == "ume_default"])
        all_row = next(row for row in summary if row["group_type"] == "all")
        self.assertEqual(all_row["auc_score_error"], 0.0)
        self.assertEqual(all_row["auc_inverse_score_error"], 1.0)
        self.assertEqual(all_row["best_direction"], "score_low_error")

    def test_auc_uncertainty_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_auc_uncertainty.py"
        spec = importlib.util.spec_from_file_location("analyze_auc_uncertainty", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        rows = [
            {"task": "t2i", "cluster_id": "a", "is_error": False, "ume": 0.8, "u_tok": 0.8, "u_intra": 0.8, "u_cfg": 0.0, "u_mod": 0.0},
            {"task": "t2i", "cluster_id": "b", "is_error": True, "ume": 0.1, "u_tok": 0.1, "u_intra": 0.1, "u_cfg": 0.0, "u_mod": 0.0},
            {"task": "t2i", "cluster_id": "c", "is_error": False, "ume": 0.7, "u_tok": 0.7, "u_intra": 0.7, "u_cfg": 0.0, "u_mod": 0.0},
            {"task": "t2i", "cluster_id": "d", "is_error": True, "ume": 0.2, "u_tok": 0.2, "u_intra": 0.2, "u_cfg": 0.0, "u_mod": 0.0},
        ]
        rng = __import__("random").Random(1)
        summary = module.summarize(
            rows,
            {"inverse_ume_default": module.region_metric_functions()["inverse_ume_default"]},
            level="region_token",
            rng=rng,
            bootstrap_iters=20,
            permutation_iters=20,
        )
        all_row = next(row for row in summary if row["group"] == "all")
        self.assertEqual(all_row["auc"], 1.0)
        self.assertEqual(all_row["errors"], 2)
        self.assertNotEqual(all_row["bootstrap_ci_low"], "")

    def test_auc_robustness_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_auc_robustness.py"
        spec = importlib.util.spec_from_file_location("analyze_auc_robustness", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        rows = [
            {
                "task": "t2i",
                "run_id": "run_a",
                "cluster_id": "a",
                "is_error": False,
                "ume": 0.8,
                "u_tok": 0.8,
                "u_intra": 0.8,
                "u_cfg": 0.0,
                "u_mod": 0.0,
            },
            {
                "task": "t2i",
                "run_id": "run_a",
                "cluster_id": "b",
                "is_error": True,
                "ume": 0.1,
                "u_tok": 0.1,
                "u_intra": 0.1,
                "u_cfg": 0.0,
                "u_mod": 0.0,
            },
            {
                "task": "t2i",
                "run_id": "run_b",
                "cluster_id": "c",
                "is_error": False,
                "ume": 0.7,
                "u_tok": 0.7,
                "u_intra": 0.7,
                "u_cfg": 0.0,
                "u_mod": 0.0,
            },
            {
                "task": "t2i",
                "run_id": "run_b",
                "cluster_id": "d",
                "is_error": True,
                "ume": 0.2,
                "u_tok": 0.2,
                "u_intra": 0.2,
                "u_cfg": 0.0,
                "u_mod": 0.0,
            },
        ]
        metric_fns = {"inverse_ume_default": module.region_metric_functions()["inverse_ume_default"]}
        detail = module.summarize_leave_one(rows, metric_fns, level="region_token", omit_key="run_id")
        baseline = next(row for row in detail if row["omit_type"] == "none" and row["group"] == "all")
        self.assertEqual(baseline["auc"], 1.0)
        summary = module.summarize_variation(detail)
        all_summary = next(row for row in summary if row["group"] == "all")
        self.assertEqual(all_summary["baseline_auc"], 1.0)
        self.assertEqual(all_summary["leave_one_count"], 2)
        self.assertEqual(all_summary["max_abs_delta"], 0.0)

    def test_prompt_cluster_effect_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_prompt_cluster_effects.py"
        spec = importlib.util.spec_from_file_location("analyze_prompt_cluster_effects", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        self.assertEqual(module.prompt_base("case_r4"), "case")
        self.assertEqual(module.prompt_base("case"), "case")
        rows = [
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "case_r1",
                "prompt_base": "case",
                "prompt_group": "t2i::run::case",
                "is_error": False,
                "visual_mean_ume": 0.8,
            },
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "case_r2",
                "prompt_base": "case",
                "prompt_group": "t2i::run::case",
                "is_error": True,
                "visual_mean_ume": 0.2,
            },
            {
                "task": "t2i",
                "run_id": "run",
                "sample_id": "other",
                "prompt_base": "other",
                "prompt_group": "t2i::run::other",
                "is_error": False,
                "visual_mean_ume": 0.7,
            },
        ]
        collapsed = module.collapsed_prompt_rows(rows, {"visual_mean_ume": lambda row: float(row["visual_mean_ume"])})
        by_group = {row["prompt_base"]: row for row in collapsed}
        self.assertEqual(by_group["case"]["samples_or_tokens"], 2)
        self.assertTrue(by_group["case"]["is_error"])
        self.assertAlmostEqual(by_group["case"]["visual_mean_ume"], 0.5)

    def test_token_extreme_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_token_extremes.py"
        spec = importlib.util.spec_from_file_location("analyze_token_extremes", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        rows = [
            {
                "task": "t2i",
                "sample_id": "case",
                "trace_file": "case_entropy.jsonl",
                "step": 1,
                "ume": 0.1,
                "u_cfg": 0.9,
                "is_error": True,
            },
            {
                "task": "t2i",
                "sample_id": "case",
                "trace_file": "case_entropy.jsonl",
                "step": 2,
                "ume": 0.8,
                "u_cfg": 0.1,
                "is_error": False,
            },
            {
                "task": "t2i",
                "sample_id": "other",
                "trace_file": "other_entropy.jsonl",
                "step": 3,
                "ume": 0.2,
                "u_cfg": 0.7,
                "is_error": True,
            },
            {
                "task": "x2i",
                "sample_id": "ref",
                "trace_file": "ref_entropy.jsonl",
                "step": 4,
                "ume": 0.6,
                "u_cfg": 0.2,
                "is_error": False,
            },
        ]
        thresholds = module.threshold_row(rows)
        self.assertEqual(thresholds["visual_tokens"], 4)
        self.assertEqual(thresholds["error_tokens"], 2)
        self.assertAlmostEqual(thresholds["ume_q25"], 0.2)
        self.assertEqual(thresholds["low_ume_error_tokens"], 2)
        self.assertEqual(thresholds["high_cfg_error_tokens"], 2)

        high_ume = module.extreme_rows(rows, metric="ume", descending=True, top_k=2)
        self.assertEqual([row["sample_id"] for row in high_ume], ["case", "ref"])
        low_error = module.extreme_rows(rows, metric="ume", descending=False, top_k=1, only_errors=True)
        self.assertEqual(low_error[0]["sample_id"], "case")

        summary = module.sample_summary_rows(rows, low_ume_threshold=0.2, high_cfg_threshold=0.7)
        by_sample = {row["sample_id"]: row for row in summary}
        self.assertEqual(by_sample["case"]["visual_tokens"], 2)
        self.assertEqual(by_sample["case"]["low_ume_error_tokens"], 1)
        self.assertEqual(by_sample["other"]["high_cfg_error_tokens"], 1)

    def test_token_case_sheet_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_token_case_sheets.py"
        spec = importlib.util.spec_from_file_location("build_token_case_sheets", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "decoded" / "case_image_00.png"
            image_path.parent.mkdir()
            Image.new("RGB", (32, 48), "white").save(image_path)
            sample_index = root / "sample_index.csv"
            sample_index.write_text(
                "task,run_id,sample_id,decoded,decoded_image\n"
                "t2i,run,case,True,decoded/case_image_00.png\n",
                encoding="utf-8",
            )
            lookup = module.decoded_image_lookup(sample_index, root)
            self.assertEqual(lookup[("t2i", "case")], image_path)

            rows = [
                {"rank": "1", "task": "t2i", "sample_id": "case", "visual_row": "1", "visual_col": "1", "ume": "0.1", "u_cfg": "0.9"},
                {"rank": "2", "task": "t2i", "sample_id": "case", "visual_row": "2", "visual_col": "0", "ume": "0.2", "u_cfg": "0.1"},
            ]
            self.assertEqual(module.grid_shape(rows, 32, 48), (3, 2))
            out = root / "overlay.png"
            info = module.draw_overlay(image_path, rows, out, color=(255, 0, 0), max_tokens=2)
            self.assertTrue(out.exists())
            self.assertEqual(info["grid_rows"], 3)
            self.assertEqual(info["grid_cols"], 2)
            self.assertEqual(info["highlighted_tokens"], 2)

    def test_error_type_strata_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_error_type_strata.py"
        spec = importlib.util.spec_from_file_location("analyze_error_type_strata", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        self.assertEqual(module.split_labels("a;b; "), ["a", "b"])
        rows = [
            {"task": "t2i", "sample_id": "bad1", "is_error": True, "error_type": "text;shape", "ume": 0.1, "u_tok": 0.1, "u_cfg": 0.8},
            {"task": "t2i", "sample_id": "bad2", "is_error": True, "error_type": "text", "ume": 0.2, "u_tok": 0.2, "u_cfg": 0.7},
            {"task": "t2i", "sample_id": "ok1", "is_error": False, "error_type": "", "ume": 0.8, "u_tok": 0.8, "u_cfg": 0.1},
            {"task": "t2i", "sample_id": "ok2", "is_error": False, "error_type": "", "ume": 0.7, "u_tok": 0.7, "u_cfg": 0.2},
        ]
        thresholds = {"ume_q25": 0.2, "u_cfg_q75": 0.7}
        summary = module.summarize_stratum(
            rows,
            label_key="error_type",
            label="text",
            group="task",
            task="t2i",
            thresholds=thresholds,
        )
        self.assertEqual(summary["positive_tokens"], 2)
        self.assertEqual(summary["negative_tokens"], 2)
        self.assertEqual(summary["positive_samples"], 2)
        self.assertEqual(summary["low_ume_tokens"], 2)
        self.assertEqual(summary["high_cfg_tokens"], 2)
        self.assertEqual(summary["auc_inverse_ume_default"], 1.0)
        self.assertEqual(summary["auc_u_cfg"], 1.0)

        all_rows = module.summarize_all(rows, label_key="error_type", min_positive_tokens=1, thresholds=thresholds)
        labels = {row["label"] for row in all_rows}
        self.assertIn("text", labels)
        self.assertIn("shape", labels)

    def test_boundary_entropy_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_boundary_entropy.py"
        spec = importlib.util.spec_from_file_location("analyze_boundary_entropy", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        records = [
            {"task": "t2i", "sample_id": "case", "step": 0, "token_type": "structure", "token_text": "<|image token|>", "segment": "image_header", "ume": 0.0, "u_mod": 0.0, "u_tok": 0.0, "u_cfg": 0.0},
            {"task": "t2i", "sample_id": "case", "step": 1, "token_type": "visual", "token_text": "v1", "ume": 0.2, "u_cfg": 0.1, "is_error": False},
            {"task": "t2i", "sample_id": "case", "step": 2, "token_type": "visual", "token_text": "v2", "ume": 0.4, "u_cfg": 0.3, "is_error": True},
            {"task": "t2i", "sample_id": "case", "step": 3, "token_type": "structure", "token_text": "<|extra_200|>", "segment": "visual", "ume": 0.0, "u_mod": 0.0, "u_tok": 0.0, "u_cfg": 0.0},
            {"task": "t2i", "sample_id": "case", "step": 4, "token_type": "visual", "token_text": "v3", "ume": 0.8, "u_cfg": 0.2, "is_error": True},
        ]
        events = module.event_rows_for_trace(Path("case_entropy.jsonl"), records, window=2)
        by_event = {row["event"]: row for row in events}
        self.assertIn("image_token", by_event)
        self.assertIn("row_break", by_event)
        self.assertAlmostEqual(by_event["image_token"]["next_mean_ume"], 0.3)
        self.assertAlmostEqual(by_event["row_break"]["prev_mean_ume"], 0.3)
        self.assertAlmostEqual(by_event["row_break"]["next_mean_ume"], 0.8)
        self.assertEqual(by_event["row_break"]["prev_error_tokens"], 1)

        summary = module.summarize_event_groups(events)
        self.assertTrue(any(row["event"] == "row_break" for row in summary))

        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.jsonl"
            labels_path.write_text(
                json.dumps({"task": "t2i", "sample_id": "case", "include_in_analysis": True, "is_error": True}) + "\n",
                encoding="utf-8",
            )
            sample_rows = module.sample_summary_rows(events, labels_path)
        self.assertEqual(len(sample_rows), 1)
        self.assertTrue(sample_rows[0]["is_error"])
        auc_rows = module.summarize_sample_auc(sample_rows)
        self.assertTrue(any(row["metric"] == "row_head_mean_ume" for row in auc_rows))

    def test_real_corpus_audit_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_real_ume_corpus.py"
        spec = importlib.util.spec_from_file_location("audit_real_ume_corpus", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        sample_rows = [
            {"task": "x2i", "run_id": "run_a", "sample_id": "000", "decoded": "True", "image_complete": "True", "visual_tokens": "2"},
            {"task": "x2i", "run_id": "run_b", "sample_id": "000", "decoded": "False", "image_complete": "False", "visual_tokens": "1"},
            {"task": "t2i", "run_id": "run_c", "sample_id": "case", "decoded": "True", "image_complete": "True", "visual_tokens": "2"},
        ]
        labels = [
            {"task": "x2i", "run_id": "run_a", "sample_id": "000", "include_in_analysis": True, "is_error": True},
            {"task": "t2i", "run_id": "run_c", "sample_id": "case", "include_in_analysis": True, "is_error": False},
        ]
        task_rows = {row["task"]: row for row in module.summarize_task_rows(sample_rows, labels)}
        self.assertEqual(task_rows["x2i"]["samples"], 2)
        self.assertEqual(task_rows["x2i"]["manual_labels"], 1)
        audit = module.collect_label_audit(sample_rows, labels)
        self.assertEqual(audit["matched_labels"], 2)
        self.assertEqual(len(audit["sample_missing_label"]), 1)
        duplicates = module.find_duplicate_short_keys(sample_rows)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["sample_id"], "000")

        records = [
            {"task": "x2i", "sample_id": "000", "token_type": "visual", "is_error": True, "manual_region_label": "bad"},
            {"task": "x2i", "sample_id": "000", "token_type": "visual", "is_error": False, "manual_region_label": ""},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "traces"
            trace_dir.mkdir()
            write_jsonl(records, trace_dir / "x2i_failed_000_entropy.jsonl")
            region = module.collect_region_trace_audit(trace_dir, labels)
        self.assertEqual(region["trace_files"], 1)
        self.assertEqual(region["visual_tokens"], 2)
        self.assertEqual(region["error_tokens"], 1)
        self.assertEqual(region["sample_rows"][0]["matched_label_candidates"], 1)

    def test_phase1_findings_report_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_phase1_findings_report.py"
        spec = importlib.util.spec_from_file_location("build_phase1_findings_report", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        self.assertEqual(module.fmt("0.123456"), "0.1235")
        rows = module.md_table(["a", "b"], [[1, "x"]])
        self.assertIn("| a | b |", rows[0])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus_audit").mkdir()
            (root / "region_labeled_analysis" / "tables").mkdir(parents=True)
            (root / "auc_uncertainty").mkdir()
            (root / "auc_robustness").mkdir()
            (root / "corpus_audit" / "task_coverage.csv").write_text(
                "task,runs,samples,decoded,complete_images,manual_labels,binary_labels,error_labels,visual_tokens\n"
                "t2i,1,2,2,2,2,2,1,20\n"
                "x2i,1,2,2,2,2,2,1,20\n",
                encoding="utf-8",
            )
            (root / "manual_sample_judgment_summary.csv").write_text(
                "group,samples,errors,error_rate,mean_quality,auc_visual_mean_ume_error,auc_visual_p90_ume_error,auc_visual_mean_u_cfg_error\n"
                "all,4,2,0.5,0.5,0.4,0.3,0.6\n"
                "t2i,2,1,0.5,0.5,0.4,0.3,0.6\n"
                "x2i,2,1,0.5,0.5,0.4,0.3,0.6\n",
                encoding="utf-8",
            )
            (root / "region_labeled_analysis" / "tables" / "hallucination_diagnostics.csv").write_text(
                "task,labeled_tokens,error_tokens,error_rate,mean_ume_error,mean_ume_correct,auc_ume_error,false_conf_tokens,false_conf_rate\n"
                "t2i,10,2,0.2,0.1,0.3,0.2,1,0.5\n"
                "x2i,10,2,0.2,0.4,0.3,0.7,1,0.5\n",
                encoding="utf-8",
            )
            (root / "auc_uncertainty" / "auc_uncertainty.csv").write_text(
                "level,group,metric,rows,errors,auc,bootstrap_ci_low,bootstrap_ci_high,permutation_p_auc_ge_observed\n"
                "region_token,all,ume_default,20,4,0.3,0.2,0.4,1.0\n"
                "region_token,all,inverse_ume_default,20,4,0.7,0.6,0.8,0.01\n"
                "region_token,all,u_cfg,20,4,0.6,0.5,0.7,0.01\n"
                "region_token,all,inverse_u_tok,20,4,0.7,0.6,0.8,0.01\n"
                "region_token,t2i,inverse_best_t2i_mix,10,2,0.7,0.6,0.8,0.01\n"
                "region_token,x2i,u_cfg,10,2,0.8,0.7,0.9,0.01\n"
                "sample,all,inverse_p90_best_all_mix,4,2,0.7,0.5,0.9,0.05\n"
                "sample,t2i,inverse_p90_u_cfg,2,1,1.0,1.0,1.0,0.01\n"
                "sample,x2i,inverse_mean_ume_default,2,1,0.5,0.0,1.0,0.5\n",
                encoding="utf-8",
            )
            (root / "auc_robustness" / "auc_robustness_summary.csv").write_text(
                "level,group,metric,omit_type,baseline_auc,min_auc,max_auc,max_abs_delta,worst_omitted_id\n"
                "region_token,all,inverse_u_tok,run_id,0.7,0.6,0.8,0.1,run\n"
                "region_token,all,inverse_ume_default,run_id,0.7,0.6,0.8,0.1,run\n"
                "region_token,all,u_cfg,run_id,0.6,0.5,0.7,0.1,run\n"
                "region_token,t2i,inverse_u_tok,run_id,0.7,0.6,0.8,0.1,run\n"
                "region_token,x2i,u_cfg,run_id,0.8,0.7,0.9,0.1,run\n"
                "sample,all,inverse_p90_best_all_mix,run_id,0.7,0.6,0.8,0.1,run\n"
                "sample,t2i,inverse_p90_u_cfg,run_id,1.0,1.0,1.0,0.0,run\n"
                "sample,x2i,inverse_p90_ume_default,run_id,0.5,0.0,1.0,0.5,run\n",
                encoding="utf-8",
            )
            (root / "phase1_figures").mkdir()
            (root / "phase1_figures" / "region_auc_summary.svg").write_text("<svg></svg>", encoding="utf-8")
            report = module.build_report(root)
        self.assertIn("Phase-1 Findings Report", report)
        self.assertIn("Executive Summary", report)
        self.assertIn("region_token", report)
        self.assertIn("phase1_figures/region_auc_summary.svg", report)

    def test_reproducibility_manifest_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_reproducibility_manifest.py"
        spec = importlib.util.spec_from_file_location("build_reproducibility_manifest", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel_path in module.KEY_ARTIFACTS:
                path = root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                text = "fixture\n"
                if rel_path == "phase1_findings_report.md":
                    text = "Synthetic fixtures are used only for code regression tests\n"
                path.write_text(text, encoding="utf-8")

            (root / "real_ume_sample_index.csv").write_text(
                "task,run_id,sample_id,decoded,image_complete\n"
                "t2i,run,case,True,True\n",
                encoding="utf-8",
            )
            (root / "manual_sample_judgments.jsonl").write_text(
                json.dumps({"task": "t2i", "run_id": "run", "sample_id": "case", "include_in_analysis": True, "is_error": False}) + "\n",
                encoding="utf-8",
            )
            (root / "corpus_audit" / "task_coverage.csv").write_text(
                "task,runs,samples,decoded,complete_images,manual_labels,binary_labels,error_labels,visual_tokens\n"
                "t2i,1,1,1,1,1,1,0,1\n",
                encoding="utf-8",
            )
            region_dir = root / "region_labeled_entropy_traces"
            region_dir.mkdir()
            (region_dir / "case_entropy.jsonl").write_text(
                json.dumps({"token_type": "visual", "is_error": False, "ume": 0.2}) + "\n",
                encoding="utf-8",
            )

            artifacts = module.artifact_rows(root, module.KEY_ARTIFACTS)
            corpus = module.corpus_snapshot(root)
            validations = module.validation_rows(root, corpus, artifacts)

        self.assertEqual(corpus["sample_rows"], 1)
        self.assertEqual(corpus["decoded_samples"], 1)
        self.assertEqual(corpus["complete_images"], 1)
        self.assertEqual(corpus["binary_manual_labels"], 1)
        self.assertEqual(corpus["region_labeled_trace_files"], 1)
        self.assertEqual(corpus["region_visual_tokens"], 1)
        self.assertTrue(all(row["exists"] for row in artifacts))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in artifacts))
        self.assertTrue(all(row["passed"] for row in validations))

    def test_phase1_coverage_plan_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_phase1_coverage_plan.py"
        spec = importlib.util.spec_from_file_location("build_phase1_coverage_plan", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        sample_rows = [
            {"task": "t2i", "run_id": "real_t2i_boundary4", "sample_id": "case", "decoded": "True", "image_complete": "True"},
            {"task": "x2i", "run_id": "real_x2i_ref_patch4", "sample_id": "edit", "decoded": "True", "image_complete": "True"},
        ]
        labels = [
            {"task": "t2i", "run_id": "real_t2i_boundary4", "sample_id": "case", "include_in_analysis": True, "is_error": False},
            {"task": "x2i", "run_id": "real_x2i_ref_patch4", "sample_id": "edit", "include_in_analysis": True, "is_error": True},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            hf_cache = Path(tmp) / "hf"
            (hf_cache / "models--BAAI--Emu3.5" / "refs").mkdir(parents=True)
            (hf_cache / "models--BAAI--Emu3.5" / "refs" / "main").write_text("partial", encoding="utf-8")
            self.assertFalse(module.model_cache_has_snapshot(hf_cache, "models--BAAI--Emu3.5"))
            progress = module.progress_rows(sample_rows, labels, hf_cache)
            buckets = module.bucket_rows(sample_rows, labels)
            actions = module.next_action_rows(progress)

        by_plan = {row["plan_item"]: row for row in progress}
        self.assertEqual(by_plan["t2i_attribute_constraints"]["samples"], 1)
        self.assertEqual(by_plan["x2i_reference_preservation"]["errors"], 1)
        self.assertIn("not cached", by_plan["interleaved_story"]["blocker"])
        self.assertTrue(any(row["bucket"] == "t2i_boundary" for row in buckets))
        self.assertTrue(any(row["priority"] == "blocked" for row in actions))

    def test_phase1_figure_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_phase1_figures.py"
        spec = importlib.util.spec_from_file_location("build_phase1_figures", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        uncertainty = [
            {
                "level": "region_token",
                "group": "all",
                "metric": "ume_default",
                "auc": "0.3",
                "bootstrap_ci_low": "0.2",
                "bootstrap_ci_high": "0.4",
            },
            {
                "level": "region_token",
                "group": "all",
                "metric": "inverse_ume_default",
                "auc": "0.7",
                "bootstrap_ci_low": "0.6",
                "bootstrap_ci_high": "0.8",
            },
        ]
        selected = module.select_rows(uncertainty, [("region_token", "all", "inverse_ume_default")])
        self.assertEqual(len(selected), 1)
        self.assertEqual(module.metric_label(selected[0]), "all / inverse default UME")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "auc.svg"
            module.auc_bar_svg(selected, "AUC", "Synthetic fixture for SVG code path.", out)
            text = out.read_text(encoding="utf-8")
        self.assertIn("<svg", text)
        self.assertIn("inverse default UME", text)
        self.assertIn("0.700", text)

    def test_auc_uncertainty_grouped_auc_helpers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_auc_uncertainty.py"
        spec = importlib.util.spec_from_file_location("analyze_auc_uncertainty", script_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        scores = [0.1, 0.2, 0.2, 0.9]
        labels = [0, 1, 0, 1]
        groups = module.score_groups(scores)
        self.assertAlmostEqual(module.auc_from_score_groups(groups, labels), module.compute_auc(scores, labels))

        weights = [1.0, 2.0, 1.0, 1.0]
        replicated_scores = [0.1, 0.2, 0.2, 0.2, 0.9]
        replicated_labels = [0, 1, 1, 0, 1]
        self.assertAlmostEqual(
            module.auc_from_score_groups(groups, labels, weights),
            module.compute_auc(replicated_scores, replicated_labels),
        )
        state = module.sorted_score_state(scores)
        self.assertAlmostEqual(module.auc_from_sorted_state(state, labels), module.compute_auc(scores, labels))
        self.assertAlmostEqual(
            module.auc_from_sorted_state(state, labels, weights),
            module.compute_auc(replicated_scores, replicated_labels),
        )


if __name__ == "__main__":
    unittest.main()
