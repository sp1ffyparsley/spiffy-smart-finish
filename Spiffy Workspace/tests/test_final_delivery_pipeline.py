from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import workflow.final_delivery_pipeline as final_delivery_pipeline
import workflow.creative_delivery_adapters as creative_delivery_adapters
from workflow.final_delivery_pipeline import (
    PLAN_SCHEMA,
    build_final_delivery_plan,
    build_worker_command,
    profiles_for_delivery_set,
    request_delivery_options,
    request_sources,
    resolve_brand_namespace,
    run_final_delivery,
    sleep_system,
    validate_final_delivery_result,
)


class FinalDeliveryPipelineTests(unittest.TestCase):
    def test_inspect_existing_failed_job_is_read_only_and_explains_resume(self) -> None:
        import contextlib
        import io
        from tools import run_spiffy_final_delivery as cli

        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            target = jobs / "failed-parent"
            target.mkdir(parents=True)
            path = target / "final_delivery_result.json"
            payload = {
                "schema": final_delivery_pipeline.RESULT_SCHEMA,
                "job_id": "failed-parent",
                "state": "failed", "attempt_id": "attempt-first",
                "source_count": 1,
                "sources": [{"source_index": 1, "state": "failed",
                             "child_job_id": "old-child"}],
                "errors": [{"code": "child_failed", "message": "Native model crash"}],
                "final_outputs": [],
            }
            path.write_text(json.dumps(payload), encoding="utf8")
            original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            original_files = sorted(p.relative_to(jobs).as_posix() for p in jobs.rglob("*") if p.is_file())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(["--jobs-root", str(jobs), "--inspect-job", "failed-parent"])
            self.assertEqual(code, 0)
            shown = json.loads(output.getvalue())
            self.assertTrue(shown["inspection_only"])
            self.assertEqual(shown["sources"][0]["child_job_id"], "old-child")
            self.assertIn("--job-id failed-parent", shown["recovery"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(
                sorted(p.relative_to(jobs).as_posix() for p in jobs.rglob("*") if p.is_file()),
                original_files)
            with self.assertRaises(SystemExit):
                cli.main(["--jobs-root", str(jobs), "--inspect-job", "failed-parent",
                          "--execute"])
            with self.assertRaises((ValueError, SystemExit)):
                cli.main(["--jobs-root", str(jobs), "--inspect-job", "../escape"])

    def test_failed_parent_retry_passes_same_source_child_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "edit.mov"
            source.write_bytes(b"original edit stays unchanged")
            commands = []

            def fail_child(command, child_dir, *_args, **_kwargs):
                commands.append(list(command))
                (child_dir / "temp").mkdir(parents=True, exist_ok=True)
                (child_dir / "progress.json").write_text(json.dumps({
                    "job_id": child_dir.name, "worker_name": "video_upscale",
                    "state": "error"}), encoding="utf8")
                (child_dir / "manifest.json").write_text(json.dumps({
                    "worker": "video_upscale"}), encoding="utf8")
                return 5

            kwargs = dict(jobs_root=root / "jobs", job_id="resumable-parent",
                          output_root=root / "delivery", profile_ids=["native_master"],
                          execute=True)
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=fail_child):
                code, previous = run_final_delivery([source], **kwargs)
            self.assertEqual(code, 1)
            self.assertEqual(previous["state"], "failed")
            previous_child = previous["sources"][0]["child_job_id"]
            self.assertTrue(previous_child)

            def second_fail(command, *_args, **_kwargs):
                commands.append(list(command))
                return 7

            with patch.object(final_delivery_pipeline, "_run_child", side_effect=second_fail):
                code, current = run_final_delivery([source], **kwargs)
            self.assertEqual(code, 1)
            self.assertEqual(current["state"], "failed")
            self.assertIn("--resume-job-id", commands[-1])
            self.assertEqual(commands[-1][commands[-1].index("--resume-job-id") + 1],
                             previous_child)
            self.assertIn("--reconcile-completed-sources", commands[-1])
            self.assertEqual(current["sources"][0]["resumed_from_child_job_id"], previous_child)
            self.assertEqual(source.read_bytes(), b"original edit stays unchanged")

    def test_failed_parent_changed_source_or_plan_must_not_use_old_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "edit.mov"
            source.write_bytes(b"original version")
            commands = []
            def fail_child(command, child_dir, *_args, **_kwargs):
                commands.append(list(command))
                (child_dir / "temp").mkdir(parents=True, exist_ok=True)
                (child_dir / "progress.json").write_text(json.dumps({
                    "job_id": child_dir.name, "worker_name": "video_upscale",
                    "state": "error"}), encoding="utf8")
                (child_dir / "manifest.json").write_text(json.dumps({
                    "worker": "video_upscale"}), encoding="utf8")
                return 5
            kwargs = dict(jobs_root=root / "jobs", job_id="changed-parent",
                          output_root=root / "delivery", profile_ids=["native_master"],
                          execute=True)
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=fail_child):
                run_final_delivery([source], **kwargs)
                source.write_bytes(b"changed version -- must rerender")
                run_final_delivery([source], **kwargs)
            self.assertNotIn("--resume-job-id", commands[-1])
            self.assertNotIn("--reconcile-completed-sources", commands[-1])

    def test_terminated_child_process_marks_stale_running_progress_as_crashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "jobs" / "terminated-child"
            child.mkdir(parents=True)
            original = {"job_id": child.name, "worker_name": "video_upscale",
                        "state": "running", "write_seq": 4}
            (child / "progress.json").write_text(json.dumps(original), encoding="utf8")
            final_delivery_pipeline._mark_exited_child_crashed(child, return_code=3221225477)
            changed = json.loads((child / "progress.json").read_text(encoding="utf8"))
            self.assertEqual(changed["state"], "crashed")
            self.assertEqual(changed["write_seq"], 5)
            self.assertEqual(changed["exit_code"], 3221225477)
            self.assertFalse(changed["can_cancel"])
            self.assertEqual(changed["worker_name"], "video_upscale")

    def test_real_child_process_exits_unclean_and_resume_progress_is_terminal(self) -> None:
        import sys
        import video_upscale as worker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            parent = final_delivery_pipeline.ProgressWriter(
                jobs, "parent", "Real child crash", "final_delivery")
            child = final_delivery_pipeline.ProgressWriter(
                jobs, "abrupt-child", "Owned worker", "video_upscale")
            child.write_manifest({"worker": "video_upscale"})
            child.update(state="running", phase="native_model")
            code = final_delivery_pipeline._run_child(
                [sys.executable, "-c", "raise SystemExit(9)"], child.job_dir,
                parent, 1, 1, attempt_started_ns=time.time_ns())
            self.assertEqual(code, 9)
            progress = json.loads((child.job_dir / "progress.json").read_text(encoding="utf8"))
            self.assertEqual(progress["state"], "crashed")
            self.assertEqual(progress["exit_code"], 9)
            self.assertEqual(worker.validated_resume_temp_dir(
                jobs, child.job_id, current_job_id="new-child"),
                child.temp_dir.resolve())

    def test_second_live_process_cannot_supersede_running_parent_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            writer = final_delivery_pipeline.ProgressWriter(
                jobs, "same-parent", "Running parent", "final_delivery")
            result_path = writer.job_dir / "final_delivery_result.json"
            claimed = {
                "schema": final_delivery_pipeline.RESULT_SCHEMA,
                "attempt_id": "attempt-original", "job_id": "same-parent",
                "state": "running",
                "attempt_owner": final_delivery_pipeline._current_attempt_owner(),
            }
            final_delivery_pipeline._claim_current_attempt(result_path, writer, claimed)
            with self.assertRaisesRegex(RuntimeError, "already_running"):
                final_delivery_pipeline._claim_current_attempt(
                    result_path, writer,
                    {**claimed, "attempt_id": "attempt-collision"})
            self.assertEqual(json.loads(result_path.read_text(encoding="utf8"))["attempt_id"],
                             "attempt-original")

    def test_orphaned_parent_with_exited_running_child_recovers_verified_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "edit.mov"
            source.write_bytes(b"unmodified source")
            jobs = root / "jobs"
            parent_id = "orphaned-parent"
            child_id = "orphaned-child"
            child_dir = jobs / child_id
            (child_dir / "temp").mkdir(parents=True)
            (child_dir / "progress.json").write_text(json.dumps({
                "job_id": child_id, "worker_name": "video_upscale", "state": "running",
                "write_seq": 9}), encoding="utf8")
            (child_dir / "manifest.json").write_text(json.dumps({
                "worker": "video_upscale", "worker_pid": 99999999}), encoding="utf8")
            plan = build_final_delivery_plan(
                source.resolve(), root / "delivery", profile_ids=["native_master"])
            owner = {"pid": 99999999, "create_time": 1.0}
            parent_dir = jobs / parent_id
            parent_dir.mkdir()
            (parent_dir / "final_delivery_result.json").write_text(json.dumps({
                "schema": final_delivery_pipeline.RESULT_SCHEMA,
                "job_id": parent_id, "attempt_id": "dead-attempt",
                "state": "running", "attempt_owner": owner,
                "sources": [{
                    "source_index": 1, "source_path": str(source.resolve()),
                    "source_identity": final_delivery_pipeline._stable_file_identity(source),
                    "plan_fingerprint": final_delivery_pipeline._plan_fingerprint(plan),
                    "child_job_id": child_id, "state": "running",
                }],
            }), encoding="utf8")
            commands = []
            def exit_again(command, *_args, **_kwargs):
                commands.append(command)
                return 6
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=exit_again):
                code, recovered = run_final_delivery(
                    [source], jobs_root=jobs, job_id=parent_id,
                    output_root=root / "delivery",
                    profile_ids=["native_master"], execute=True)
            self.assertEqual(code, 1)
            self.assertEqual(recovered["state"], "failed")
            self.assertEqual(recovered["sources"][0]["resumed_from_child_job_id"], child_id)
            self.assertIn("--resume-job-id", commands[0])
            self.assertEqual(commands[0][commands[0].index("--resume-job-id") + 1], child_id)
            child_state = json.loads((child_dir / "progress.json").read_text(encoding="utf8"))
            self.assertEqual(child_state["state"], "crashed")
            self.assertEqual(child_state["write_seq"], 10)
            self.assertEqual(source.read_bytes(), b"unmodified source")

    def test_hard_parent_exit_leaves_active_child_identity_in_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"immutable source")
            def terminated_parent(*_args, **_kwargs):
                raise SystemExit("simulate parent process termination")
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=terminated_parent):
                with self.assertRaisesRegex(SystemExit, "termination"):
                    run_final_delivery([source],
                        jobs_root=root / "jobs", job_id="terminated-parent",
                        output_root=root / "out", profile_ids=["native_master"],
                        execute=True)
            current = json.loads((root / "jobs" / "terminated-parent" /
                                  "final_delivery_result.json").read_text(encoding="utf8"))
            self.assertEqual(current["state"], "running")
            self.assertEqual(len(current["sources"]), 1)
            row = current["sources"][0]
            self.assertEqual(row["state"], "running")
            self.assertTrue(row["child_job_id"].startswith("final_delivery_child"))
            self.assertTrue(row["plan_fingerprint"])
            self.assertTrue(row["source_identity"]["sha256"])
            self.assertTrue((root / "jobs" / row["child_job_id"] / "delivery_plan.json").is_file())

    def test_parent_crash_after_completed_child_adopts_only_revalidated_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable media")
            child_run, completed_states = self._successful_child(source.resolve())
            def completed_then_parent_exits(*args, **kwargs):
                child_run(*args, **kwargs)
                child_dir = args[1]
                manifest_path = child_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf8"))
                manifest["worker_pid"] = 99999999  # Exited child owner.
                manifest_path.write_text(json.dumps(manifest), encoding="utf8")
                raise SystemExit("killed after child committed media")
            kwargs = dict(
                jobs_root=root / "jobs", job_id="orphan-after-child",
                output_root=root / "delivery",
                profile_ids=["native_master"], execute=True)
            with patch.object(final_delivery_pipeline, "_run_child",
                              side_effect=completed_then_parent_exits), patch.object(
                              final_delivery_pipeline, "validated_completed_source_states",
                              side_effect=completed_states):
                with self.assertRaises(SystemExit):
                    run_final_delivery([source], **kwargs)
            result_path = root / "jobs" / "orphan-after-child" / "final_delivery_result.json"
            old = json.loads(result_path.read_text(encoding="utf8"))
            completed_child = old["sources"][0]["child_job_id"]
            old["attempt_owner"] = {"pid": 99999999, "create_time": 1.0}
            result_path.write_text(json.dumps(old), encoding="utf8")
            with patch.object(final_delivery_pipeline, "_run_child",
                              side_effect=AssertionError("verified media should be reused")) as child, patch.object(
                              final_delivery_pipeline, "validated_completed_source_states",
                              side_effect=completed_states):
                code, recovered = run_final_delivery([source], **kwargs)
            self.assertEqual(code, 0)
            self.assertEqual(recovered["state"], "passed")
            self.assertEqual(recovered["sources"][0]["child_job_id"], completed_child)
            self.assertTrue(recovered["sources"][0]["media_reused"])
            child.assert_not_called()
            self.assertEqual(source.read_bytes(), b"stable media")

    def test_still_running_child_blocks_orphan_takeover_without_overwriting_progress(self) -> None:
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            child_id = "active-child"
            child = jobs / child_id
            child.mkdir(parents=True)
            running = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(8)", "--job-id", child_id],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                (child / "manifest.json").write_text(
                    json.dumps({"worker": "video_upscale", "worker_pid": running.pid}),
                    encoding="utf8")
                target = jobs / "orphan-parent"
                target.mkdir()
                receipt = target / "final_delivery_result.json"
                receipt.write_text(json.dumps({
                    "schema": final_delivery_pipeline.RESULT_SCHEMA,
                    "job_id": "orphan-parent", "state": "running",
                    "attempt_id": "old-abandoned-attempt",
                    "attempt_owner": {"pid": 99999999, "create_time": 1.0},
                    "sources": [{"child_job_id": child_id, "state": "running"}],
                }), encoding="utf8")
                original = receipt.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "child_already_running"):
                    run_final_delivery(
                        [root / "source.mov"], jobs_root=jobs,
                        job_id="orphan-parent", output_root=root / "out",
                        profile_ids=["native_master"], execute=True)
                self.assertEqual(receipt.read_bytes(), original)
                self.assertFalse((target / "progress.json").exists())
            finally:
                running.terminate()
                running.wait(timeout=5)

    def _successful_child(self, source: Path):
        def run(command, child_dir, parent, file_index, file_total, *, attempt_started_ns):
            plan = json.loads((child_dir / "delivery_plan.json").read_text(encoding="utf-8"))
            output = Path(plan["variants"][0]["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"verified final delivery bytes")
            (child_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "worker": "video_upscale",
                        "delivery_plan": plan,
                        "final_outputs": [str(output)],
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )
            return 0

        def completed_states(manifest):
            plan = manifest["delivery_plan"]
            variant = dict(plan["variants"][0])
            if (
                int(variant.get("width") or 0) == 0
                and int(variant.get("height") or 0) == 0
                and str(variant.get("layout") or "") == "keep_source_aspect"
            ):
                variant["width"] = 1920
                variant["height"] = 1080
            output = Path(variant["output_path"])
            completion = {
                "fingerprint": "completion-fixture",
                "contract": {"variant": variant},
                "output": {"path": str(output)},
                "probe": {
                    "width": int(variant.get("width") or 1920),
                    "height": int(variant.get("height") or 1080),
                    "fps": 30.0,
                    "duration": 1.0,
                },
            }
            return {
                "states": [
                    {
                        "matched_source_key": final_delivery_pipeline._path_key(source),
                        "resolved_source": {
                            "variants": [variant],
                            "profile_output_map": dict(plan["profile_output_map"]),
                            "physical_output_count": int(plan["physical_output_count"]),
                        },
                        "output_paths": [str(output)],
                        "completion_rows": [completion],
                    }
                ],
                "stale_or_unproven_completed_sources": [],
                "completed_source_row_count": 1,
                "validated_current_completed_source_count": 1,
            }

        return run, completed_states

    def test_known_brand_namespaces_are_stable_and_distinct(self) -> None:
        self.assertEqual(resolve_brand_namespace("Spiffy Scenes"), "spiffy.scenes")
        self.assertEqual(resolve_brand_namespace("Spiffy Scene"), "spiffy.scenes")
        self.assertEqual(resolve_brand_namespace("spiffy_mma"), "spiffy.mma")
        self.assertEqual(resolve_brand_namespace("Spiffy Parsley"), "spiffy.parsley")
        self.assertEqual(resolve_brand_namespace("Luna's Music"), "spiffy.luna")
        self.assertEqual(resolve_brand_namespace("Luna.Music"), "spiffy.luna")
        self.assertEqual(resolve_brand_namespace("Spiffy Clips HQ"), "spiffy.clips")
        self.assertNotEqual(resolve_brand_namespace("Spiffy Clips"), resolve_brand_namespace("Spiffy Luna"))

    def test_chatgpt_is_not_accepted_as_a_delivery_brand(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a Spiffy delivery brand"):
            resolve_brand_namespace("ChatGPT")

    def test_invisible_watermark_is_opt_in_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_final_delivery_plan(
                root / "source.mov",
                root / "delivery",
                profile_ids=["native_master"],
            )
            self.assertTrue(plan["variants"])
            self.assertTrue(all(row["watermark"] is False for row in plan["variants"]))
        options = request_delivery_options({"final_delivery": {"profile_ids": ["native_master"]}})
        self.assertFalse(options["watermark"])

    def test_all_social_profiles_deduplicate_to_four_physical_geometries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_final_delivery_plan(
                root / "source.mov",
                root / "delivery",
                profile_ids=profiles_for_delivery_set("all_social"),
                brand="spiffy-scenes",
            )
        self.assertEqual(plan["schema"], PLAN_SCHEMA)
        self.assertEqual(plan["physical_output_count"], 4)
        self.assertEqual(len(plan["requested_profile_ids"]), 11)
        self.assertEqual(
            {row["watermark_identifier"] for row in plan["variants"]},
            {"native-master", "portrait-4x5", "vertical-short-9x16", "horizontal-16x9"},
        )
        self.assertEqual(
            plan["profile_output_map"]["tiktok_vertical"],
            plan["profile_output_map"]["snapchat_story"],
        )
        self.assertEqual(
            plan["profile_output_map"]["youtube_horizontal"],
            plan["profile_output_map"]["dailymotion_horizontal"],
        )

    def test_default_layout_fits_with_black_and_never_stretches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_final_delivery_plan(
                root / "source.mov",
                root / "delivery",
                profile_ids=["native_master", "generic_vertical_short"],
            )
        native, vertical = plan["variants"]
        self.assertEqual(native["layout"], "keep_source_aspect")
        self.assertEqual((native["width"], native["height"]), (0, 0))
        self.assertEqual(vertical["layout"], "fit_into_frame")
        self.assertEqual(vertical["background"], "black")
        self.assertTrue(plan["policy"]["never_stretch"])

    def test_smart_finish_master_is_explicit_and_preserves_legacy_native_master(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = build_final_delivery_plan(
                root / "source.mov",
                root / "delivery",
                profile_ids=["native_master"],
            )
            smart_finish = build_final_delivery_plan(
                root / "source.mov",
                root / "delivery",
                profile_ids=profiles_for_delivery_set("smart_finish"),
            )

        self.assertEqual(profiles_for_delivery_set("smart_finish"), ["smart_finish_master"])
        legacy_variant = legacy["variants"][0]
        smart_variant = smart_finish["variants"][0]
        self.assertEqual(legacy_variant["physical_name"], "native_master")
        self.assertEqual(legacy_variant["layout"], "keep_source_aspect")
        self.assertEqual((legacy_variant["width"], legacy_variant["height"]), (0, 0))
        self.assertNotIn("geometry_policy", legacy_variant)
        self.assertEqual(smart_finish["enhance_preset"], "4k_60fps")
        self.assertEqual(smart_variant["physical_name"], "smart_finish_master")
        self.assertEqual(smart_variant["geometry_policy"], "source_display")
        self.assertEqual(smart_variant["layout"], "fit_into_frame")
        self.assertEqual((smart_variant["width"], smart_variant["height"]), (0, 0))
        self.assertNotEqual(legacy_variant["output_path"], smart_variant["output_path"])
        resolved = final_delivery_pipeline._variant_signature(
            smart_variant,
            resolved_dimensions={"width": 1920, "height": 1080},
        )
        self.assertEqual((resolved["width"], resolved["height"]), (1920, 1080))
        self.assertEqual(resolved["geometry_policy"], "source_display")

    def test_crop_is_explicit_not_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_final_delivery_plan(
                root / "source.mov",
                root / "delivery",
                profile_ids=["generic_vertical_short"],
                layout_policy="fill_crop",
            )
        self.assertEqual(plan["variants"][0]["layout"], "fill_crop_center")
        self.assertIn("_crop.mp4", plan["variants"][0]["output_path"])

    def test_request_adapter_accepts_new_and_legacy_autoexport_manifests(self) -> None:
        payload = {
            "version": 2,
            "expected_files": [r"C:\render\one.mov", r"C:\render\two.mov"],
            "final_delivery": {
                "delivery_set": "short_form",
                "brand": "spiffy-clips",
                "watermark": True,
            },
        }
        self.assertEqual(len(request_sources(payload)), 2)
        options = request_delivery_options(payload)
        self.assertEqual(options["brand"], "spiffy-clips")
        self.assertEqual(options["profile_ids"], ["native_master", "spiffy_4x5", "generic_vertical_short"])
        smart_options = request_delivery_options({"final_delivery": {"delivery_set": "smart_finish"}})
        self.assertEqual(smart_options["profile_ids"], ["smart_finish_master"])

    def test_worker_command_lets_plan_own_watermark_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = build_worker_command(
                jobs_root=root,
                child_job_id="child",
                input_list=root / "inputs.txt",
                plan_path=root / "plan.json",
            )
        self.assertEqual(command[command.index("--watermark") + 1], "off")
        self.assertEqual(command[command.index("--delivery-plan") + 1], str(root / "plan.json"))

    def test_dry_run_writes_real_job_plan_without_requiring_source_to_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, result = run_final_delivery(
                [root / "pending_from_premiere.mov"],
                jobs_root=root / "jobs",
                job_id="dry-run",
                output_root=root / "delivery",
                profile_ids=["native_master"],
                execute=False,
            )
            job_dir = root / "jobs" / "dry-run"
            child_dir = root / "jobs" / result["sources"][0]["child_job_id"]
            self.assertEqual(code, 0)
            self.assertEqual(result["state"], "planned")
            self.assertTrue((job_dir / "final_delivery_result.json").is_file())
            self.assertTrue((child_dir / "delivery_plan.json").is_file())
            self.assertEqual(json.loads((job_dir / "progress.json").read_text())["state"], "done")

    def test_execute_fails_closed_when_child_exits_zero_without_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable source")
            stale_child = root / "jobs" / "delivery_001"
            stale_child.mkdir(parents=True)
            (stale_child / "manifest.json").write_text(
                json.dumps({"worker": "video_upscale", "final_outputs": ["stale.mp4"]}),
                encoding="utf-8",
            )
            with patch.object(final_delivery_pipeline, "_run_child", return_value=0):
                code, result = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="delivery",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )

            self.assertEqual(code, 1)
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["final_outputs"], [])
            self.assertNotEqual(result["sources"][0]["child_job_id"], "delivery_001")
            self.assertIn(
                "child_manifest_missing_or_malformed",
                result["sources"][0]["validation_issues"],
            )
            self.assertEqual(
                validate_final_delivery_result(root / "jobs" / "delivery")["status"],
                "FAIL",
            )

    def test_execute_emits_exact_current_output_proofs_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable source bytes")
            child_run, completed_states = self._successful_child(source.resolve())
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=completed_states,
            ):
                code, result = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="proofs",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )

            self.assertEqual(code, 0)
            self.assertEqual(result["state"], "passed")
            self.assertEqual(len(result["final_outputs"]), 1)
            self.assertEqual(len(result["final_output_proofs"]), 1)
            proof = result["final_output_proofs"][0]
            output = Path(proof["path"])
            self.assertEqual(proof["bytes"], output.stat().st_size)
            self.assertEqual((proof["width"], proof["height"]), (1920, 1080))
            self.assertEqual(validate_final_delivery_result(root / "jobs" / "proofs")["status"], "PASS")
            downstream = creative_delivery_adapters._verify_delivery_receipt(
                output,
                root / "jobs" / "proofs" / "final_delivery_result.json",
            )
            self.assertEqual(downstream["status"], "VERIFIED")
            self.assertEqual(downstream["authority"], "final_delivery_hash_and_size")
            output.write_bytes(b"mutated after delivery authority")
            validation = validate_final_delivery_result(root / "jobs" / "proofs")
            self.assertEqual(validation["status"], "FAIL")
            self.assertIn("final_delivery_output_identity_mismatch", validation["issues"])

    def test_execute_rejects_child_watermark_identifier_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable source bytes")
            child_run, completed_states = self._successful_child(source.resolve())

            def drifted_completed_states(manifest):
                payload = completed_states(manifest)
                payload["states"][0]["resolved_source"]["variants"][0]["watermark_identifier"] = "wrong-video-type"
                return payload

            with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=drifted_completed_states,
            ):
                code, result = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="identifier-drift",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    watermark=True,
                    execute=True,
                )

            self.assertEqual(code, 1)
            self.assertEqual(result["state"], "failed")
            self.assertIn(
                "child_delivery_variant_binding_mismatch",
                result["sources"][0]["validation_issues"],
            )

    def test_current_parent_revalidates_signed_receipt_and_legacy_child_reference(self) -> None:
        from workflow import video_provenance_watermark as watermark

        secret = b"parent-test-only-receipt-secret-32"
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source.mov"
                source.write_bytes(b"stable source bytes")
                child_run, completed_states = self._successful_child(source.resolve())

                def signed_completed_states(manifest):
                    payload = completed_states(manifest)
                    completion = payload["states"][0]["completion_rows"][0]
                    variant = completion["contract"]["variant"]
                    output = Path(variant["output_path"])
                    receipt_path = output.with_suffix(".mp4.videoseal.json")
                    receipt = {
                        "schema": watermark.RECEIPT_SCHEMA,
                        "asset_id": "parent-fixture-asset",
                        "namespace": variant["namespace"],
                        "identifier": variant["watermark_identifier"],
                        "delivery_profile_id": variant["profile_id"],
                        "source_identity": {"identity_key": "parent-fixture-source"},
                        "output_full_identity": watermark._full_export_identity_witness(output),
                        "gates": {"payload_recovery": "pass", "scheduled_sample_coverage": "pass"},
                    }
                    receipt["payload_digest_sha256"] = hashlib.sha256(watermark.derive_payload(
                        secret, asset_id=receipt["asset_id"], source_identity_key="parent-fixture-source",
                        namespace=receipt["namespace"], identifier=receipt["identifier"],
                    )).hexdigest()
                    receipt["receipt_hmac_sha256"] = watermark.sign_receipt(receipt, secret)
                    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                    completion["watermark_result"] = {
                        "status": "pass", "output": str(output), "receipt": str(receipt_path),
                        "asset_id": receipt["asset_id"], "namespace": receipt["namespace"],
                        "identifier": receipt["identifier"], "delivery_profile_id": receipt["delivery_profile_id"],
                    }
                    manifest["completed_delivery_outputs"] = [completion]
                    # Persist the fake child's legacy receipt reference for the
                    # explicit backward-compatible public-result readback case.
                    for path in (root / "jobs").glob("final_delivery_child_*/manifest.json"):
                        path.write_text(json.dumps(manifest), encoding="utf-8")
                    return payload

                with patch.object(watermark, "load_or_create_secret", return_value=secret), patch.object(
                    watermark, "load_runtime", side_effect=AssertionError("readback must not load a model")
                ), patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                    final_delivery_pipeline, "validated_completed_source_states", side_effect=signed_completed_states
                ):
                    code, result = run_final_delivery(
                        [source], jobs_root=root / "jobs", job_id="signed-parent",
                        output_root=root / "delivery", profile_ids=["native_master"], watermark=True, execute=True,
                    )
                    self.assertEqual(code, 0)
                    result_path = root / "jobs" / "signed-parent" / "final_delivery_result.json"
                    if legacy:
                        for proof in result["final_output_proofs"]:
                            proof.pop("watermark_enabled", None)
                            proof.pop("watermark_receipt", None)
                        result_path.write_text(json.dumps(result), encoding="utf-8")
                    else:
                        self.assertTrue(result["final_output_proofs"][0]["watermark_enabled"])
                        self.assertEqual(result["final_output_proofs"][0]["watermark_receipt"]["asset_id"], "parent-fixture-asset")
                    self.assertEqual(validate_final_delivery_result(result_path)["status"], "PASS")
                    Path(result["final_outputs"][0]).with_suffix(".mp4.videoseal.json").unlink()
                    validation = validate_final_delivery_result(result_path)
                    self.assertEqual(validation["status"], "FAIL")
                    self.assertIn("final_delivery_watermark_receipt_missing_or_invalid", validation["issues"])

    def test_execute_rejects_nonpositive_or_missing_child_dimensions(self) -> None:
        cases = [
            {"width": 0, "height": 1080},
            {"width": 1920, "height": 0},
            {"width": 0, "height": 0},
            {"width": -1920, "height": 1080},
            {"width": 1920, "height": -1080},
            {"height": 1080},
            {"width": 1920},
            {"width": None, "height": 1080},
            {"width": True, "height": 1080},
            {"width": 1920.5, "height": 1080},
            {"width": "invalid", "height": 1080},
        ]
        for dimensions in cases:
            with self.subTest(dimensions=dimensions), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source.mov"
                source.write_bytes(b"stable source bytes")
                child_run, completed_states = self._successful_child(source.resolve())

                def invalid_completed_states(manifest):
                    payload = completed_states(manifest)
                    variant = payload["states"][0]["resolved_source"]["variants"][0]
                    variant.pop("width", None)
                    variant.pop("height", None)
                    variant.update(dimensions)
                    return payload

                with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                    final_delivery_pipeline,
                    "validated_completed_source_states",
                    side_effect=invalid_completed_states,
                ):
                    code, result = run_final_delivery(
                        [source],
                        jobs_root=root / "jobs",
                        job_id="invalid-dimensions",
                        output_root=root / "delivery",
                        profile_ids=["native_master"],
                        execute=True,
                    )

                self.assertEqual(code, 1)
                self.assertEqual(result["state"], "failed")
                self.assertEqual(result["final_outputs"], [])
                self.assertIn(
                    "child_delivery_variant_dimensions_invalid",
                    result["sources"][0]["validation_issues"],
                )

    def test_execute_preserves_explicit_dimensions_and_rejects_drift(self) -> None:
        for drift in (False, True):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source.mov"
                source.write_bytes(b"stable source bytes")
                child_run, completed_states = self._successful_child(source.resolve())

                def fixed_completed_states(manifest):
                    payload = completed_states(manifest)
                    if drift:
                        payload["states"][0]["resolved_source"]["variants"][0]["width"] += 2
                    return payload

                with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                    final_delivery_pipeline,
                    "validated_completed_source_states",
                    side_effect=fixed_completed_states,
                ):
                    code, result = run_final_delivery(
                        [source],
                        jobs_root=root / "jobs",
                        job_id="fixed-dimensions",
                        output_root=root / "delivery",
                        profile_ids=["generic_vertical_short"],
                        execute=True,
                    )

                if drift:
                    self.assertEqual(code, 1)
                    self.assertEqual(result["state"], "failed")
                    self.assertEqual(result["final_outputs"], [])
                    self.assertIn(
                        "child_delivery_variant_binding_mismatch",
                        result["sources"][0]["validation_issues"],
                    )
                else:
                    self.assertEqual(code, 0)
                    self.assertEqual(result["state"], "passed")
                    proof = result["final_output_proofs"][0]
                    self.assertEqual((proof["width"], proof["height"]), (1080, 1920))

    def test_current_result_never_falls_back_to_archived_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable source")
            child_run, completed_states = self._successful_child(source.resolve())
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=completed_states,
            ):
                code, passed = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="history",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )
            self.assertEqual(code, 0)
            code, planned = run_final_delivery(
                [source],
                jobs_root=root / "jobs",
                job_id="history",
                output_root=root / "delivery",
                profile_ids=["native_master"],
                execute=False,
            )
            self.assertEqual(code, 0)
            history = root / "jobs" / "history" / "history" / f"final_delivery_result_{passed['attempt_id']}.json"
            self.assertTrue(history.is_file())
            self.assertEqual(planned["state"], "planned")
            self.assertEqual(validate_final_delivery_result(root / "jobs" / "history")["status"], "FAIL")

            current = root / "jobs" / "history" / "final_delivery_result.json"
            current.write_text('{"schema":', encoding="utf-8")
            validation = validate_final_delivery_result(root / "jobs" / "history")
            self.assertEqual(validation["status"], "FAIL")
            self.assertIn("final_delivery_result_missing_or_malformed", validation["issues"])

    def test_metadata_publication_retry_reuses_exact_validated_media_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable source")
            child_run, completed_states = self._successful_child(source.resolve())
            real_commit = final_delivery_pipeline._commit_current_result

            def fail_pass_commit(result_path, writer, result):
                if result.get("state") == "passed":
                    raise OSError("simulated parent metadata publication failure")
                return real_commit(result_path, writer, result)

            with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run) as first_child, patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=completed_states,
            ), patch.object(
                final_delivery_pipeline,
                "_commit_current_result",
                side_effect=fail_pass_commit,
            ):
                code, failed = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="retry",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )
            self.assertEqual(code, 1)
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(first_child.call_count, 1)

            with patch.object(
                final_delivery_pipeline,
                "_run_child",
                side_effect=AssertionError("exact validated media should be reused"),
            ) as retry_child, patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=completed_states,
            ):
                code, retried = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="retry",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )
            self.assertEqual(code, 0)
            self.assertEqual(retried["state"], "passed")
            self.assertTrue(retried["sources"][0]["media_reused"])
            retry_child.assert_not_called()
            self.assertEqual(validate_final_delivery_result(root / "jobs" / "retry")["status"], "PASS")

    def test_changed_source_bytes_prevent_media_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"source generation one")
            child_run, completed_states = self._successful_child(source.resolve())
            real_commit = final_delivery_pipeline._commit_current_result

            def fail_pass_commit(result_path, writer, result):
                if result.get("state") == "passed":
                    raise OSError("simulated parent metadata publication failure")
                return real_commit(result_path, writer, result)

            with patch.object(final_delivery_pipeline, "_run_child", side_effect=child_run), patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=completed_states,
            ), patch.object(
                final_delivery_pipeline,
                "_commit_current_result",
                side_effect=fail_pass_commit,
            ):
                first_code, _ = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="source-change",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )
            self.assertEqual(first_code, 1)
            source.write_bytes(b"source generation two changed")

            with patch.object(final_delivery_pipeline, "_run_child", return_value=9) as retry_child:
                second_code, second = run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id="source-change",
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )
            self.assertEqual(second_code, 1)
            self.assertFalse(second["sources"][0]["media_reused"])
            retry_child.assert_called_once()

    def test_explicit_recipe_retry_reuses_only_identical_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable recipe test source")
            child_run, completed_states = self._successful_child(source.resolve())

            def recipe_child(command, child_dir, *args, **kwargs):
                code = child_run(command, child_dir, *args, **kwargs)
                path = child_dir / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["enhancement_recipe"] = manifest["delivery_plan"]["enhancement_recipe"]
                path.write_text(json.dumps(manifest), encoding="utf-8")
                return code

            def recipe_states(manifest):
                states = completed_states(manifest)
                for row in states["states"][0]["completion_rows"]:
                    row["contract"]["enhancement_recipe"] = manifest["enhancement_recipe"]
                return states

            kwargs = dict(jobs_root=root / "jobs", job_id="recipe-retry", output_root=root / "out",
                          profile_ids=["native_master"], execute=True)
            enhancement = {"preset_id": "ss_4k", "overrides": {"noise": 0.1}}
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=recipe_child) as run, patch.object(
                final_delivery_pipeline, "validated_completed_source_states", side_effect=recipe_states
            ):
                code, first = run_final_delivery([source], enhancement=enhancement, **kwargs)
                self.assertEqual(code, 0)
                self.assertEqual(run.call_count, 1)
                code, same = run_final_delivery([source], enhancement=enhancement, **kwargs)
                self.assertEqual(code, 0)
                self.assertTrue(same["sources"][0]["media_reused"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(first["enhancement_recipe"], same["final_output_proofs"][0]["enhancement_recipe"])
            with patch.object(final_delivery_pipeline, "_run_child", return_value=9) as run:
                code, changed = run_final_delivery([source], enhancement={"preset_id": "ss_4k", "overrides": {"noise": 0.2}}, **kwargs)
                self.assertEqual(code, 1)
                self.assertFalse(changed["sources"][0]["media_reused"])
                run.assert_called_once()

    def test_exact_cadence_retry_reuses_same_rate_and_rejects_changed_rate(self) -> None:
        """Receipt/dispatch contract test, not fabricated native image proof."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable cadence contract fixture")
            child_run, completed_states = self._successful_child(source.resolve())

            def rate_child(command, child_dir, *args, **kwargs):
                code = child_run(command, child_dir, *args, **kwargs)
                path = child_dir / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf8"))
                manifest["frame_rate"] = manifest["delivery_plan"]["frame_rate"]
                path.write_text(json.dumps(manifest), encoding="utf8")
                return code

            def rate_states(manifest):
                states = completed_states(manifest)
                for row in states["states"][0]["completion_rows"]:
                    row["contract"]["frame_rate"] = manifest["frame_rate"]
                    row["probe"]["fps"] = 90.0
                    row["probe"]["fps_ratio"] = "90/1"
                return states

            kwargs = dict(jobs_root=root / "jobs", job_id="rate-retry", output_root=root / "out",
                          profile_ids=["native_master"], execute=True)
            with patch.object(final_delivery_pipeline, "_run_child", side_effect=rate_child) as run, patch.object(
                final_delivery_pipeline, "validated_completed_source_states", side_effect=rate_states
            ):
                code, _ = run_final_delivery([source], frame_rate={"mode": "cfr", "fps": 90}, **kwargs)
                self.assertEqual(code, 0)
                code, same = run_final_delivery([source], frame_rate={"mode": "cfr", "fps": "180/2"}, **kwargs)
                self.assertEqual(code, 0)
                self.assertTrue(same["sources"][0]["media_reused"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(same["final_output_proofs"][0]["frame_rate"], {"mode": "cfr", "fps": "90/1"})
            with patch.object(final_delivery_pipeline, "_run_child", return_value=9) as run:
                code, changed = run_final_delivery([source], frame_rate={"mode": "cfr", "fps": 120}, **kwargs)
                self.assertEqual(code, 1)
                self.assertFalse(changed["sources"][0]["media_reused"])
                run.assert_called_once()

    def test_same_target_concurrent_attempts_serialize_child_media_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mov"
            source.write_bytes(b"stable concurrent source")
            child_run, completed_states = self._successful_child(source.resolve())
            active = 0
            max_active = 0
            state_lock = threading.Lock()

            def slow_child(*args, **kwargs):
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.15)
                    return child_run(*args, **kwargs)
                finally:
                    with state_lock:
                        active -= 1

            def run(job_id: str):
                return run_final_delivery(
                    [source],
                    jobs_root=root / "jobs",
                    job_id=job_id,
                    output_root=root / "delivery",
                    profile_ids=["native_master"],
                    execute=True,
                )

            with patch.object(
                final_delivery_pipeline,
                "_run_child",
                side_effect=slow_child,
            ), patch.object(
                final_delivery_pipeline,
                "validated_completed_source_states",
                side_effect=completed_states,
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(run, ("concurrent-a", "concurrent-b")))

            self.assertEqual(max_active, 1)
            self.assertEqual([code for code, _result in results], [0, 0])
            lanes = {
                result["sources"][0]["delivery_target_lane"]["lane"]
                for _code, result in results
            }
            self.assertEqual(len(lanes), 1)

    def test_superseded_same_job_attempt_cannot_publish_current_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            writer_a = final_delivery_pipeline.ProgressWriter(
                jobs,
                "same-job",
                "Spiffy Final Delivery",
                "final_delivery",
            )
            result_path = writer_a.job_dir / "final_delivery_result.json"
            attempt_a = {
                "schema": final_delivery_pipeline.RESULT_SCHEMA,
                "job_id": "same-job",
                "attempt_id": "attempt-a",
                "state": "running",
                "errors": [],
            }
            final_delivery_pipeline._claim_current_attempt(
                result_path,
                writer_a,
                attempt_a,
            )
            writer_b = final_delivery_pipeline.ProgressWriter(
                jobs,
                "same-job",
                "Spiffy Final Delivery",
                "final_delivery",
            )
            attempt_b = {
                "schema": final_delivery_pipeline.RESULT_SCHEMA,
                "job_id": "same-job",
                "attempt_id": "attempt-b",
                "state": "running",
                "errors": [],
            }
            final_delivery_pipeline._claim_current_attempt(
                result_path,
                writer_b,
                attempt_b,
            )

            with self.assertRaisesRegex(RuntimeError, "final_delivery_attempt_superseded"):
                final_delivery_pipeline._commit_current_result(
                    result_path,
                    writer_a,
                    {
                        **attempt_a,
                        "state": "passed",
                        "final_outputs": ["should-not-publish.mp4"],
                    },
                )
            current = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(current["attempt_id"], "attempt-b")
            self.assertEqual(current["state"], "running")

    def test_sleep_system_routes_through_platform_adapter(self) -> None:
        with patch.object(final_delivery_pipeline, "platform_name", return_value="macos"), patch.object(
            final_delivery_pipeline.subprocess, "run"
        ) as run:
            sleep_system()

        run.assert_called_once_with(["pmset", "sleepnow"], check=True)

    def test_sleep_system_rejects_unsupported_platform_without_spawning(self) -> None:
        with patch.object(final_delivery_pipeline, "platform_name", return_value="linux"), patch.object(
            final_delivery_pipeline.subprocess, "run"
        ) as run, self.assertRaisesRegex(RuntimeError, "Windows and macOS"):
            sleep_system()

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
