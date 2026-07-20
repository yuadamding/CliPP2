from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "runs" / "benchmark_sim_v2_cuda.py"
SPEC = importlib.util.spec_from_file_location("sim_v2_cuda_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_parser_normalizes_backend_and_one_lambda_override() -> None:
    args = benchmark.parse_args(
        [
            "--package-parent",
            "/tmp/source",
            "--source-label",
            "revised",
            "--backend",
            "quotient-workset",
            "--strict-backend",
            "--max-unique-lambdas",
            "1",
            "--output-json",
            "/tmp/result.json",
            "/tmp/case.tsv",
        ]
    )

    assert args.backend == "quotient_workset"
    assert args.strict_backend is True
    assert args.max_unique_lambdas == 1
    assert args.cases == [Path("/tmp/case.tsv")]


def test_strict_backend_rejects_auto() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--package-parent",
                "/tmp/source",
                "--source-label",
                "auto",
                "--backend",
                "auto",
                "--strict-backend",
                "--output-json",
                "/tmp/result.json",
                "/tmp/case.tsv",
            ]
        )


def _quotient_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "tumor_id": "case-a",
        "selection_step": 0,
        "lambda": 0.5,
        "lambda_applicable": True,
        "candidate_pool_source": "raw_fused_lambda_path",
        "estimator_role": "raw_fused_lambda_path",
        "device": "cuda:0",
        "inner_iterations": 12,
        "admm_iterations": 0,
        "inner_solver": "quotient_workset_complete_graph",
        "inner_backend": "quotient_workset_complete_graph",
        "backend_iterations": 12,
        "quotient_iterations": 4,
        "workset_iterations": 8,
        "workset_expansions": 2,
        "streamed_edge_passes": 3,
        "dense_iterations": 0,
        "fallback_reason": "",
    }
    record.update(updates)
    return record


def test_backend_observations_capture_each_counter_and_pass_strict_audit() -> None:
    records = [_quotient_record(), _quotient_record(selection_step=1)]
    records[1]["lambda"] = 1.0

    observations = benchmark._backend_observations(
        records, requested_backend="quotient_workset"
    )
    audit = benchmark._audit_backend_observations(
        observations, requested_backend="quotient_workset"
    )

    assert len(observations) == 2
    assert observations[0]["actual_backend_family"] == "quotient_workset"
    assert observations[0]["quotient_iterations"] == 4
    assert observations[0]["workset_iterations"] == 8
    assert observations[0]["workset_expansions"] == 2
    assert observations[0]["dense_iterations"] == 0
    assert observations[0]["fallback_reason"] == ""
    assert observations[0]["partition_signature"] is None
    assert observations[0]["missing_backend_columns"] == []
    assert audit == {
        "requested_backend": "quotient_workset",
        "expected_device_type": "cuda",
        "applicable_rows": 2,
        "passed": True,
        "violations": [],
    }


def test_backend_observation_preserves_correctness_and_timing_evidence() -> None:
    evidence = {
        "partition_signature": "2:a,b|1:c",
        "partition_hash": "partition-sha256",
        "cluster_sizes": "2,1",
        "penalized_objective": 10.25,
        "raw_objective": 10.25,
        "loglik": -8.0,
        "raw_loglik": -8.0,
        "summary_loglik": -7.75,
        "bic": 22.0,
        "classic_bic": 21.5,
        "extended_bic": 23.0,
        "partition_icl": -12.0,
        "bic_loglik": -7.5,
        "bic_n_clusters": 2,
        "n_clusters": 2,
        "fixed_objective_kkt_residual": 2.5e-5,
        "full_kkt_tolerance": 5e-4,
        "raw_kkt_eligible": True,
        "bic_selection_eligible": True,
        "candidate_elapsed_seconds": 1.25,
        "raw_fit_elapsed_seconds": 0.75,
        "bic_refit_elapsed_seconds": 0.5,
    }

    observation = benchmark._backend_observations(
        [_quotient_record(**evidence)], requested_backend="quotient_workset"
    )[0]

    assert {column: observation[column] for column in evidence} == evidence
    assert observation["missing_backend_columns"] == []
    assert (
        benchmark._audit_backend_observations(
            [observation], requested_backend="quotient_workset"
        )["passed"]
        is True
    )


def test_strict_audit_reports_actual_backend_and_fallback_violations() -> None:
    observations = benchmark._backend_observations(
        [
            _quotient_record(
                inner_backend="admm_complete_graph",
                inner_solver="admm_complete_graph",
                dense_iterations=9,
                fallback_reason="dense_current_device_after_quotient_attempt",
            )
        ],
        requested_backend="quotient_workset",
    )

    audit = benchmark._audit_backend_observations(
        observations, requested_backend="quotient_workset"
    )

    assert audit["passed"] is False
    assert {violation["kind"] for violation in audit["violations"]} == {
        "actual_backend_family",
        "fallback_reason",
    }


@pytest.mark.parametrize("recorded_device", ["cpu", None, "mps:0"])
def test_strict_audit_rejects_non_cuda_or_missing_device(
    recorded_device: object,
) -> None:
    observations = benchmark._backend_observations(
        [_quotient_record(device=recorded_device)],
        requested_backend="quotient_workset",
    )

    audit = benchmark._audit_backend_observations(
        observations,
        requested_backend="quotient_workset",
        expected_device_type="cuda:0",
    )

    assert audit["passed"] is False
    assert audit["expected_device_type"] == "cuda"
    assert [violation["kind"] for violation in audit["violations"]] == [
        "actual_device_type"
    ]
    assert audit["violations"][0]["recorded_device"] == recorded_device


def test_lambda_budget_counts_unique_values_not_retries() -> None:
    observations = benchmark._backend_observations(
        [
            _quotient_record(selection_step=0),
            _quotient_record(selection_step=1),
        ],
        requested_backend="quotient_workset",
    )
    for row in observations:
        row["lambda"] = 0.5

    audit = benchmark._lambda_budget_audit(observations, max_unique_lambdas=1)

    assert audit["observed_unique_lambdas"] == [0.5]
    assert audit["observed_unique_lambda_count"] == 1
    assert audit["passed"] is True


def test_temporary_lambda_budget_restores_all_bindings() -> None:
    runner = SimpleNamespace(PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS=12)
    config = SimpleNamespace(PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS=12)

    with benchmark._temporary_max_unique_lambdas((runner, config), 1) as metadata:
        assert runner.PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS == 1
        assert config.PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS == 1
        assert metadata["effective"] == 1
        assert metadata["override_requested"] is True

    assert runner.PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS == 12
    assert config.PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS == 12


def test_fit_options_drop_fields_unsupported_by_an_archived_source() -> None:
    class ArchivedFitOptions:
        __dataclass_fields__ = {
            "lambda_value": object(),
            "device": object(),
            "dtype": object(),
        }

        def __init__(self, **values: object) -> None:
            self.values = values

    core_model = SimpleNamespace(FitOptions=ArchivedFitOptions)
    args = benchmark.parse_args(
        [
            "--package-parent",
            "/tmp/source",
            "--source-label",
            "archived",
            "--backend",
            "dense",
            "--output-json",
            "/tmp/result.json",
            "/tmp/case.tsv",
        ]
    )

    compatibility = benchmark._fit_options_compatibility(core_model, args)
    options = benchmark._fit_options(core_model, args)

    assert options.values == {
        "lambda_value": 0.0,
        "device": "cuda",
        "dtype": "float64",
    }
    assert "inner_backend" in compatibility["unsupported_option_names"]
    assert "dense_fallback_policy" in compatibility["unsupported_option_names"]


def test_python_source_manifest_hashes_relative_paths_and_bytes_only(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "CliPP2"
    core = package_root / "core"
    ignored_runs = package_root / "runs"
    core.mkdir(parents=True)
    ignored_runs.mkdir()
    sources = {
        "__init__.py": b"VERSION = 1\n",
        "core/__init__.py": b"from .model import VALUE\n",
        "core/model.py": b"VALUE = 7\n",
    }
    for relative_path, content in sources.items():
        path = package_root / relative_path
        path.write_bytes(content)
    (ignored_runs / "benchmark.py").write_text("ignored = True\n", encoding="utf-8")

    manifest = benchmark._python_source_manifest(package_root)
    expected = hashlib.sha256()
    for relative_path, content in sorted(sources.items()):
        path_bytes = relative_path.encode("utf-8")
        expected.update(len(path_bytes).to_bytes(8, byteorder="big", signed=False))
        expected.update(path_bytes)
        expected.update(len(content).to_bytes(8, byteorder="big", signed=False))
        expected.update(content)

    assert manifest == {
        "algorithm": "sha256_length_prefixed_posix_relative_path_and_bytes_v1",
        "sha256": expected.hexdigest(),
        "file_count": 3,
        "total_file_bytes": sum(map(len, sources.values())),
    }
    (ignored_runs / "benchmark.py").write_text(
        "still_ignored = True\n", encoding="utf-8"
    )
    assert benchmark._python_source_manifest(package_root) == manifest
    (core / "model.py").write_text("VALUE = 8\n", encoding="utf-8")
    assert (
        benchmark._python_source_manifest(package_root)["sha256"] != manifest["sha256"]
    )


def _complete_case(case_path: Path, *, selection_step: int = 0) -> dict[str, object]:
    observations = benchmark._backend_observations(
        [_quotient_record(selection_step=selection_step)],
        requested_backend="quotient_workset",
    )
    return {
        "case_path": str(case_path),
        "case_name": case_path.stem,
        "status": "complete",
        "timing": {"wall_seconds": 1.5},
        "cuda_memory": {
            "max_memory_allocated_bytes": 100 + selection_step,
            "max_memory_reserved_bytes": 200 + selection_step,
        },
        "backend_observations": observations,
        "backend_audit": benchmark._audit_backend_observations(
            observations, requested_backend="quotient_workset"
        ),
    }


def test_progress_checkpoint_is_atomic_valid_partial_json(tmp_path: Path) -> None:
    output = tmp_path / "partial.json"
    payload = {"cases": [_complete_case(Path("case-a.tsv"))]}

    benchmark._write_progress_checkpoint(
        output_json=output,
        payload=payload,
        total_case_count=2,
        state="running",
        checkpoint_sequence=1,
        strict_backend=True,
        backend_violations=[],
        budget_failures=[],
        first_error=None,
    )

    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint["aggregate"]["case_count"] == 1
    assert checkpoint["aggregate"]["completed_case_count"] == 1
    assert checkpoint["run_status"] == {
        "active_case_path": None,
        "attempted_case_count": 1,
        "checkpoint_sequence": 1,
        "checkpointed_at_utc": checkpoint["run_status"]["checkpointed_at_utc"],
        "completed_case_count": 1,
        "failed_case_count": 0,
        "is_final": False,
        "remaining_case_count": 1,
        "state": "running",
        "total_case_count": 2,
    }
    assert checkpoint["strict_backend_result"]["passed"] is None
    assert checkpoint["strict_backend_result"]["provisional_passed"] is True
    assert "finished_at_utc" not in checkpoint
    assert list(tmp_path.glob(".partial.json.tmp-*")) == []


def _mock_run_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    interrupt_on_call: int | None = None,
) -> tuple[object, list[dict[str, object]], KeyboardInterrupt | None]:
    case_paths = [tmp_path / "case-a.tsv", tmp_path / "case-b.tsv"]
    output = tmp_path / "benchmark.json"
    args = benchmark.parse_args(
        [
            "--package-parent",
            str(tmp_path / "source"),
            "--source-label",
            "checkpoint-test",
            "--backend",
            "quotient-workset",
            "--output-json",
            str(output),
            *(str(path) for path in case_paths),
        ]
    )

    class FakeFitOptions:
        __dataclass_fields__: dict[str, object] = {}

    runner = SimpleNamespace(PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS=12)
    config = SimpleNamespace(PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS=12)
    modules = {
        "package": object(),
        "core_model": SimpleNamespace(FitOptions=FakeFitOptions),
        "pipeline": SimpleNamespace(process_one_file_bundle=object()),
        "runner_selection": runner,
        "selection_config": config,
    }
    fake_torch = object()
    monkeypatch.setattr(
        benchmark,
        "_resolve_inputs",
        lambda unused_args: (tmp_path / "source", case_paths, output),
    )
    monkeypatch.setattr(
        benchmark, "_import_isolated_source", lambda unused_parent: modules
    )
    monkeypatch.setattr(
        benchmark.importlib,
        "import_module",
        lambda name: (
            fake_torch if name == "torch" else pytest.fail(f"unexpected import {name}")
        ),
    )
    monkeypatch.setattr(
        benchmark, "_resolve_cuda_device", lambda unused_torch, unused_device: "cuda:0"
    )
    monkeypatch.setattr(
        benchmark, "_configure_reproducibility", lambda *unused, **unused_kw: None
    )
    monkeypatch.setattr(
        benchmark, "_environment_metadata", lambda *unused: {"environment": "test"}
    )
    monkeypatch.setattr(
        benchmark, "_determinism_metadata", lambda *unused, **unused_kw: {}
    )
    monkeypatch.setattr(
        benchmark, "_cuda_warmup", lambda *unused, **unused_kw: {"enabled": False}
    )
    monkeypatch.setattr(
        benchmark, "_source_metadata", lambda **unused: {"label": "checkpoint-test"}
    )

    interrupt = (
        None if interrupt_on_call is None else KeyboardInterrupt("benchmark stopped")
    )
    call_count = 0

    def fake_run_case(**values: object) -> tuple[dict[str, object], None]:
        nonlocal call_count
        current_call = call_count
        call_count += 1
        if current_call == interrupt_on_call:
            assert interrupt is not None
            raise interrupt
        return _complete_case(
            Path(values["case_path"]), selection_step=current_call
        ), None

    monkeypatch.setattr(benchmark, "_run_case", fake_run_case)
    snapshots: list[dict[str, object]] = []

    def capture_checkpoint(unused_path: Path, payload: object) -> None:
        snapshots.append(json.loads(json.dumps(benchmark._json_safe(payload))))

    monkeypatch.setattr(benchmark, "_atomic_write_json", capture_checkpoint)
    return args, snapshots, interrupt


def test_run_checkpoints_once_after_each_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, snapshots, _ = _mock_run_environment(monkeypatch, tmp_path)

    result = benchmark.run_benchmark(args)

    assert len(snapshots) == 2
    assert snapshots[0]["run_status"]["state"] == "running"
    assert snapshots[0]["run_status"]["completed_case_count"] == 1
    assert snapshots[0]["run_status"]["remaining_case_count"] == 1
    assert snapshots[0]["aggregate"]["case_count"] == 1
    assert snapshots[1]["run_status"]["state"] == "complete"
    assert snapshots[1]["run_status"]["is_final"] is True
    assert snapshots[1]["run_status"]["completed_case_count"] == 2
    assert snapshots[1]["aggregate"]["case_count"] == 2
    assert result["run_status"]["checkpoint_sequence"] == 2


def test_keyboard_interrupt_checkpoints_then_reraises_same_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, snapshots, interrupt = _mock_run_environment(
        monkeypatch, tmp_path, interrupt_on_call=1
    )
    assert interrupt is not None

    with pytest.raises(KeyboardInterrupt) as caught:
        benchmark.run_benchmark(args)

    assert caught.value is interrupt
    assert len(snapshots) == 2
    assert snapshots[0]["run_status"]["state"] == "running"
    interrupted = snapshots[1]
    assert interrupted["run_status"]["state"] == "interrupted"
    assert interrupted["run_status"]["is_final"] is True
    assert interrupted["run_status"]["attempted_case_count"] == 1
    assert interrupted["run_status"]["remaining_case_count"] == 1
    assert interrupted["run_status"]["active_case_path"].endswith("case-b.tsv")
    assert interrupted["interruption"]["type"] == "KeyboardInterrupt"
    assert interrupted["strict_backend_result"]["passed"] is None
