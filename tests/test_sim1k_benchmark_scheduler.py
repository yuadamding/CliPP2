from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "runs" / "run_clipp2sim1k_benchmark.py"
SPEC = importlib.util.spec_from_file_location("clipp2sim1k_benchmark_scheduler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


CASE_SMALL = "50_2_0.3_0.0_S1_Lm300_M2_rep0"
CASE_LARGE = "100_2_0.6_0.2_S2_Lm600_M3_rep1"
CONFIG_SHA256 = "c" * 64


def _write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_input(
    path: Path, tumor_id: str, *, mutations: int, regions: int, cna: bool
) -> None:
    header = [
        "mutation_id",
        "sample_id",
        "ref_counts",
        "alt_counts",
        "normal_cn",
        "major_cn",
        "minor_cn",
        "has_cna",
        "purity",
    ]
    rows: list[list[object]] = []
    for mutation in range(mutations):
        for region in range(regions):
            asymmetric = cna and mutation == 0
            rows.append(
                [
                    f"m{mutation + 1}",
                    f"{tumor_id}_sample{region}",
                    18,
                    2,
                    2,
                    2 if asymmetric else 1,
                    1,
                    1,
                    0.6,
                ]
            )
    _write_tsv(path, header, rows)


def _cohort(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    truth = tmp_path / "truth"
    inputs.mkdir()
    truth.mkdir()
    specifications = (
        (CASE_LARGE, 3, 2, True, [1, 1, 2]),
        (CASE_SMALL, 2, 1, False, [1, 2]),
    )
    for tumor_id, mutations, regions, cna, truth_labels in specifications:
        _write_input(
            inputs / f"{tumor_id}.tsv",
            tumor_id,
            mutations=mutations,
            regions=regions,
            cna=cna,
        )
        case_truth = truth / tumor_id
        case_truth.mkdir()
        _write_tsv(
            case_truth / "truth.txt",
            ["cluster_id"],
            [[label] for label in truth_labels],
        )
    return inputs, truth


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        device="cuda",
        outer_max_iter=8,
        inner_max_iter=30,
        tol=1e-4,
        summary_tol=1e-4,
        bic_partition_tol=1e-4,
        major_prior=0.5,
        inner_backend="dense",
        workset_max_bytes=scheduler.DEFAULT_WORKSET_MAX_BYTES,
        compressed_cache_max_bytes=scheduler.DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
        dense_fallback_policy="auto",
        workset_add_batch=scheduler.DEFAULT_WORKSET_ADD_BATCH,
        workset_max_expansions=scheduler.DEFAULT_WORKSET_MAX_EXPANSIONS,
        certificate_max_iter=scheduler.DEFAULT_CERTIFICATE_MAX_ITER,
        certificate_refinement_rounds=scheduler.DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
        certificate_column_tol_scale=scheduler.DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
        allow_heuristic_structure_splits=True,
        materialize_full_dual=False,
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        timeout_seconds=100.0,
        termination_grace_seconds=5.0,
    )


def _config() -> dict[str, object]:
    return scheduler.build_run_config(
        _args(),
        cohort_sha256="a" * 64,
        source_sha256="b" * 64,
        environment_sha256="e" * 64,
    )


def _case_context(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    inputs, truth = _cohort(tmp_path)
    manifest = scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)
    case = next(dict(row) for row in manifest["cases"] if row["tumor_id"] == CASE_SMALL)
    return inputs, truth, case, _config()


def _fake_artifacts(
    directory: Path,
    input_file: Path,
    case: dict[str, object],
    config: dict[str, object],
    *,
    device: str = "cuda",
    candidate_truth_leak: bool = False,
    candidate_evaluation_mode: str = "not_evaluated",
    exact_provenance: bool = False,
    certificate_status: str = "certified",
    certificate_scope: str = "full_original_graph",
) -> None:
    from CliPP2.io.data import load_tumor_tsv

    data = load_tumor_tsv(input_file, missing_cna_policy="error")
    tumor_id = str(case["tumor_id"])
    labels = [
        (index % int(case["true_cluster_count"])) + 1
        for index in range(data.num_mutations)
    ]
    cluster_sizes = {label: labels.count(label) for label in sorted(set(labels))}
    graph_name = "complete_adaptive_test"
    directory.mkdir(parents=True)
    _write_tsv(
        directory / f"{tumor_id}_mutation_clusters.tsv",
        ["tumor_id", "mutation_id", "cluster_label", "cluster_size"],
        [
            [tumor_id, mutation_id, label, cluster_sizes[label]]
            for mutation_id, label in zip(data.mutation_ids, labels, strict=True)
        ],
    )
    _write_tsv(
        directory / f"{tumor_id}_cluster_centers.tsv",
        ["tumor_id", "cluster_label", "cluster_size"],
        [[tumor_id, label, cluster_sizes[label]] for label in sorted(cluster_sizes)],
    )
    multiplicity_header = [
        "tumor_id",
        "mutation_id",
        "region_id",
        "cluster_label",
        "phi",
        "summary_phi",
        "major_cn",
        "minor_cn",
        "multiplicity_estimated",
        "gamma_major",
        "major_call",
        "multiplicity_call",
    ]
    multiplicity_rows: list[list[object]] = []
    for mutation_index, mutation_id in enumerate(data.mutation_ids):
        for region_index, region_id in enumerate(data.region_ids):
            estimated = bool(
                data.multiplicity_estimation_mask[mutation_index, region_index]
            )
            major = float(data.major_cn[mutation_index, region_index])
            multiplicity_rows.append(
                [
                    tumor_id,
                    mutation_id,
                    str(region_id).replace("sample", "region"),
                    labels[mutation_index],
                    0.75,
                    0.75,
                    major,
                    float(data.minor_cn[mutation_index, region_index]),
                    int(estimated),
                    0.75 if estimated else 1.0,
                    1,
                    major,
                ]
            )
    _write_tsv(
        directory / f"{tumor_id}_mutation_region_multiplicity.tsv",
        multiplicity_header,
        multiplicity_rows,
    )

    exact_columns = (
        [
            "inner_backend",
            "backend_iterations",
            "exactness_provenance_version",
            "estimator_role",
            "objective_faithful",
            "objective_spec_hash",
            "original_graph_hash",
            "certificate_problem_hash",
            "certificate_scope",
            "certificate_gradient_scope",
            "full_kkt_certified",
            "full_kkt_certificate_status",
            "full_kkt_tolerance",
        ]
        if exact_provenance
        else []
    )
    inner_solver = (
        "quotient_workset_complete_graph" if exact_provenance else "admm_complete_graph"
    )
    admm_iterations = 0 if exact_provenance else 10
    exact_values = (
        [
            inner_solver,
            10,
            scheduler.EXACTNESS_PROVENANCE_VERSION,
            "raw_fused_lambda_path",
            True,
            "objective-v1",
            "graph-v1",
            "problem-v1",
            certificate_scope,
            "observed_objective",
            True,
            certificate_status,
            5e-4,
        ]
        if exact_provenance
        else []
    )
    lambda_header = [
        "tumor_id",
        "selection_score_name",
        "selection_step",
        "lambda",
        "lambda_applicable",
        "candidate_pool_source",
        "partition_icl",
        "n_clusters",
        "fixed_objective_kkt_residual",
        "raw_kkt_eligible",
        "bic_selection_eligible",
        "eligible_for_selection",
        "inner_solver",
        "admm_iterations",
        *exact_columns,
        "graph_name",
        "num_edges",
        "tol",
        "input_data_hash",
        "lambda_search_mode",
        "lambda_path_prespecified",
        "evaluation_mode",
        "candidate_evaluation_elapsed_seconds",
        *scheduler.LAMBDA_TRUTH_COLUMNS,
        "is_selection_optimal",
        "is_selected_best_row",
    ]
    lambda_rows: list[list[object]] = []
    for step, (lambda_value, score, selected) in enumerate(
        ((1.0, 20.0, False), (2.0, 10.0, True))
    ):
        truth_values = ["" for _ in scheduler.LAMBDA_TRUTH_COLUMNS]
        if candidate_truth_leak and step == 0:
            truth_values[0] = 0.9
        lambda_rows.append(
            [
                tumor_id,
                "partition_icl",
                step,
                lambda_value,
                True,
                "raw_fused_lambda_path",
                score,
                len(cluster_sizes),
                4e-4,
                True,
                True,
                True,
                inner_solver,
                admm_iterations,
                *exact_values,
                graph_name,
                data.num_mutations * (data.num_mutations - 1) // 2,
                1e-4,
                case["input_data_hash"],
                "partition_guided_admm",
                False,
                candidate_evaluation_mode,
                0.0,
                *truth_values,
                selected,
                selected,
            ]
        )
    _write_tsv(directory / f"{tumor_id}_lambda_search.tsv", lambda_header, lambda_rows)

    asymmetric = bool((data.major_cn != data.minor_cn).any())
    estimable = bool(data.multiplicity_estimation_mask.any())
    primary_f1: object = 0.8 if asymmetric else ""
    estimable_f1: object = 0.75 if estimable else ""
    evaluation_header = [
        "ARI",
        "cp_rmse",
        "raw_cp_rmse",
        "summary_cp_rmse",
        "bic_refit_cp_rmse",
        "multiplicity_f1",
        "multiplicity_asymmetric_f1",
        "multiplicity_estimable_f1",
        "estimated_clonal_fraction",
        "true_clonal_fraction",
        "clonal_fraction_error",
        "true_clusters",
        "estimated_clusters",
        "n_eval_mutations",
        "n_filtered_mutations",
    ]
    evaluation_row = [
        0.8,
        0.02,
        0.03,
        0.02,
        0.019,
        primary_f1,
        primary_f1,
        estimable_f1,
        0.5,
        0.5,
        0.0,
        case["true_cluster_count"],
        len(cluster_sizes),
        data.num_mutations,
        0,
    ]
    _write_tsv(
        directory / f"{tumor_id}_simulation_eval.tsv",
        evaluation_header,
        [evaluation_row],
    )

    summary = {
        "tumor_id": tumor_id,
        "device": device,
        "dtype": "float64",
        "lambda_search_mode": "partition_guided_admm",
        "selection_score_name": "partition_icl",
        "lambda_path_prespecified": False,
        "evaluate_all_candidates": False,
        "inner_solver": inner_solver,
        "admm_iterations": admm_iterations,
        "selected_lambda": 2.0,
        "selected_candidate_pool_source": "raw_fused_lambda_path",
        "selected_kkt_residual": 4e-4,
        "n_clusters": len(cluster_sizes),
        "num_regions": data.num_regions,
        "num_mutations": data.num_mutations,
        "input_data_hash": case["input_data_hash"],
        "tol": 1e-4,
        "major_prior": config["major_prior"],
        "bic_df_scale": config["bic_df_scale"],
        "bic_cluster_penalty": config["bic_cluster_penalty"],
        "selection_eligible": True,
        "num_candidates_all": 2,
        "graph_name": graph_name,
        "ARI": 0.8,
        "cp_rmse": 0.02,
        "raw_cp_rmse": 0.03,
        "summary_cp_rmse": 0.02,
        "multiplicity_f1": primary_f1,
        "multiplicity_asymmetric_f1": primary_f1,
        "multiplicity_estimable_f1": estimable_f1,
        "benchmark_config_sha256": CONFIG_SHA256,
        "benchmark_source_sha256": config["source_sha256"],
        "benchmark_environment_sha256": config["environment_sha256"],
        "benchmark_cohort_sha256": config["cohort_sha256"],
        "benchmark_input_sha256": case["input_sha256"],
        "benchmark_truth_sha256": case["truth_sha256"],
    }
    if exact_provenance:
        summary.update(dict(zip(exact_columns, exact_values, strict=True)))
    _write_tsv(
        directory / f"{tumor_id}_run_summary.tsv",
        list(summary),
        [list(summary.values())],
    )


def _validation_kwargs(
    inputs: Path,
    case: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "expected_device": "cuda",
        "input_file": inputs / str(case["input_file"]),
        "expected_case": case,
        "config": config,
        "config_sha256": CONFIG_SHA256,
    }


def test_manifest_preflight_hashes_dimensions_and_truth(tmp_path: Path) -> None:
    inputs, truth = _cohort(tmp_path)
    manifest = scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)

    assert manifest["case_count"] == 2
    cases = {row["tumor_id"]: row for row in manifest["cases"]}
    assert cases[CASE_SMALL]["mutation_count"] == 2
    assert cases[CASE_LARGE]["region_count"] == 2
    assert cases[CASE_SMALL]["true_cluster_count"] == 2
    assert len(cases[CASE_SMALL]["input_data_hash"]) == 32
    assert cases[CASE_SMALL]["schedule_rank"] == 1

    _write_input(
        inputs / f"{CASE_SMALL}.tsv", CASE_SMALL, mutations=1, regions=1, cna=False
    )
    with pytest.raises(ValueError, match="Input dimensions disagree with filename"):
        scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)


def test_manifest_preflight_rejects_input_truth_mismatch(tmp_path: Path) -> None:
    inputs, truth = _cohort(tmp_path)
    (truth / CASE_SMALL).rename(truth / "unmatched_case")
    with pytest.raises(ValueError, match="Input/truth cohort mismatch"):
        scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)


def test_source_manifest_uses_canonical_package_root_and_hashes_solver_source() -> None:
    manifest = scheduler.build_source_manifest()
    paths = {str(row["path"]) for row in manifest["files"]}

    assert Path(str(manifest["repo_root"])) == scheduler.REPO_ROOT
    assert "cli.py" in paths
    assert "core/fusion/solver.py" in paths
    assert "runs/run_clipp2sim1k_benchmark.py" in paths
    assert not any(path.startswith(".venv/") for path in paths)
    assert scheduler.DEFAULT_INPUT_DIR == scheduler.REPO_ROOT.parent / "CliPP2Sim1K_TSV"
    assert (
        scheduler.DEFAULT_SIMULATION_ROOT == scheduler.REPO_ROOT.parent / "CliPP2Sim1K"
    )


def test_full_factorial_validator_requires_every_cell_exactly_once() -> None:
    cases = [
        {
            "depth": depth,
            "region_count": regions,
            "target_mutation_count": target_m,
            "cna_rate": cna,
            "purity": purity,
            "replicate": replicate,
        }
        for depth in scheduler.EXPECTED_DEPTHS
        for regions in scheduler.EXPECTED_REGION_COUNTS
        for target_m in scheduler.EXPECTED_TARGET_MUTATION_COUNTS
        for cna in scheduler.EXPECTED_CNA_RATES
        for purity, replicate in scheduler.EXPECTED_PURITY_REPLICATES
    ]
    scheduler._validate_factorial_design(cases)
    cases[-1] = dict(cases[0])
    with pytest.raises(ValueError, match="duplicate condition cells"):
        scheduler._validate_factorial_design(cases)


def test_immutable_document_and_environment_lock_refuse_resume_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    first = scheduler._write_document_and_hash(path, {"device": "cuda"}, label="config")
    assert (
        scheduler._write_document_and_hash(path, {"device": "cuda"}, label="config")
        == first
    )
    with pytest.raises(RuntimeError, match="Immutable config mismatch"):
        scheduler._write_document_and_hash(path, {"device": "cpu"}, label="config")

    first_environment = {
        "captured_at": "first",
        "packages": [{"name": "torch", "version": "1"}],
    }
    later_environment = {
        "captured_at": "later",
        "packages": [{"name": "torch", "version": "1"}],
    }
    assert scheduler.stable_environment_fingerprint(
        first_environment
    ) == scheduler.stable_environment_fingerprint(later_environment)
    changed_environment = {
        "captured_at": "later",
        "packages": [{"name": "torch", "version": "2"}],
    }
    assert scheduler.stable_environment_fingerprint(
        first_environment
    ) != scheduler.stable_environment_fingerprint(changed_environment)


def test_validated_bundle_is_promoted_atomically_and_flattened_with_copies(
    tmp_path: Path,
) -> None:
    inputs, _, case, config = _case_context(tmp_path)
    outdir = tmp_path / "run"
    stage = outdir / ".staging" / "case"
    input_file = inputs / str(case["input_file"])
    _fake_artifacts(stage, input_file, case, config)

    bundle, hashes = scheduler.promote_case_bundle(
        stage,
        outdir,
        CASE_SMALL,
        **_validation_kwargs(inputs, case, config),
    )

    assert bundle == outdir / "bundles" / CASE_SMALL
    assert not stage.exists()
    assert set(hashes) == set(scheduler.expected_artifact_names(CASE_SMALL))
    for name in scheduler.expected_artifact_names(CASE_SMALL):
        flat = outdir / "results" / name
        stored = bundle / name
        assert flat.read_bytes() == stored.read_bytes()
        assert not os.path.samefile(stored, flat)
    flat_summary = outdir / "results" / f"{CASE_SMALL}_run_summary.tsv"
    stored_summary = bundle / f"{CASE_SMALL}_run_summary.tsv"
    original = stored_summary.read_bytes()
    flat_summary.write_bytes(b"changed\n")
    assert stored_summary.read_bytes() == original


def test_artifact_validation_rejects_truth_leakage_wrong_backend_and_truncation(
    tmp_path: Path,
) -> None:
    inputs, _, case, config = _case_context(tmp_path)
    input_file = inputs / str(case["input_file"])
    kwargs = _validation_kwargs(inputs, case, config)

    leaked = tmp_path / "leaked"
    _fake_artifacts(leaked, input_file, case, config, candidate_truth_leak=True)
    with pytest.raises(ValueError, match="candidate truth metrics"):
        scheduler.validate_case_artifacts(leaked, CASE_SMALL, **kwargs)

    evaluated = tmp_path / "evaluated"
    _fake_artifacts(
        evaluated,
        input_file,
        case,
        config,
        candidate_evaluation_mode="full",
    )
    with pytest.raises(ValueError, match="truth evaluated before selection"):
        scheduler.validate_case_artifacts(evaluated, CASE_SMALL, **kwargs)

    cpu_stage = tmp_path / "cpu"
    _fake_artifacts(cpu_stage, input_file, case, config, device="cpu")
    with pytest.raises(ValueError, match="device='cpu'"):
        scheduler.validate_case_artifacts(cpu_stage, CASE_SMALL, **kwargs)

    truncated = tmp_path / "truncated"
    _fake_artifacts(truncated, input_file, case, config)
    multiplicity = truncated / f"{CASE_SMALL}_mutation_region_multiplicity.tsv"
    lines = multiplicity.read_text(encoding="utf-8").splitlines()
    multiplicity.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiplicity output.*expected 2"):
        scheduler.validate_case_artifacts(truncated, CASE_SMALL, **kwargs)


def test_artifact_validation_accepts_schema_v1_quotient_provenance_and_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, _, case, config = _case_context(tmp_path)
    config = {**config, "inner_backend": "quotient_workset"}
    input_file = inputs / str(case["input_file"])
    kwargs = _validation_kwargs(inputs, case, config)

    valid = tmp_path / "quotient-valid"
    _fake_artifacts(valid, input_file, case, config, exact_provenance=True)
    scheduler.validate_case_artifacts(valid, CASE_SMALL, **kwargs)

    invalid_status = tmp_path / "quotient-invalid-status"
    _fake_artifacts(
        invalid_status,
        input_file,
        case,
        config,
        exact_provenance=True,
        certificate_status="not_certified",
    )
    with pytest.raises(ValueError, match="unaccepted full KKT certificate status"):
        scheduler.validate_case_artifacts(invalid_status, CASE_SMALL, **kwargs)

    legacy = tmp_path / "quotient-legacy"
    _fake_artifacts(legacy, input_file, case, config)
    with pytest.raises(ValueError, match="lacks schema-v1 exactness provenance"):
        scheduler.validate_case_artifacts(legacy, CASE_SMALL, **kwargs)


def test_sqlite_state_retries_exports_and_recovery_cleans_attempt(
    tmp_path: Path,
) -> None:
    inputs, truth = _cohort(tmp_path)
    manifest = scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)
    state = scheduler.BenchmarkState(tmp_path / "run" / "state.sqlite3")
    try:
        state.initialize(manifest)
        attempt = state.start_attempt(
            CASE_SMALL,
            backend="cuda",
            log_path=tmp_path / "first.log",
            staging_path=tmp_path / "first-stage",
            timeout_seconds=12.0,
            termination_grace_seconds=3.0,
        )
        assert attempt == 1
        assert (
            state.finish_failure(
                CASE_SMALL,
                attempt,
                elapsed_seconds=1.0,
                returncode=1,
                timed_out=False,
                error_type="CalledProcessError",
                error_message="failed",
            )
            == "retry_pending"
        )
        attempt = state.start_attempt(
            CASE_SMALL,
            backend="cuda",
            log_path=tmp_path / "second.log",
            staging_path=tmp_path / "second-stage",
        )
        assert attempt == 2
        state.recover_success(
            CASE_SMALL,
            backend="cuda",
            bundle_path=tmp_path / "bundle",
            artifact_hashes={"artifact": "hash"},
        )
        scheduler.export_state(state, tmp_path / "run")
        failures = (tmp_path / "run" / "failures.tsv").read_text(encoding="utf-8")
        assert "CalledProcessError" in failures
        assert "second.log" not in failures
        second_attempt = state.connection.execute(
            "SELECT status FROM attempts WHERE tumor_id=? AND attempt=2", (CASE_SMALL,)
        ).fetchone()
        assert second_attempt["status"] == "succeeded"
    finally:
        state.close()


def test_reconcile_recovers_crash_after_atomic_bundle_rename(tmp_path: Path) -> None:
    inputs, truth, case, config = _case_context(tmp_path)
    manifest = scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)
    outdir = tmp_path / "run"
    state = scheduler.BenchmarkState(outdir / "state.sqlite3")
    try:
        state.initialize(manifest)
        state.start_attempt(
            CASE_SMALL,
            backend="cuda",
            log_path=outdir / "case.log",
            staging_path=outdir / ".staging" / "case",
        )
        stage = outdir / ".staging" / "case"
        _fake_artifacts(stage, inputs / str(case["input_file"]), case, config)
        bundle, expected_hashes = scheduler.promote_case_bundle(
            stage,
            outdir,
            CASE_SMALL,
            **_validation_kwargs(inputs, case, config),
        )

        scheduler.reconcile_bundles(
            state,
            manifest,
            outdir,
            device="cuda",
            config=config,
            config_sha256=CONFIG_SHA256,
        )

        row = next(row for row in state.case_rows() if row["tumor_id"] == CASE_SMALL)
        assert row["status"] == "succeeded"
        assert Path(row["bundle_path"]) == bundle
        assert json.loads(row["artifact_hashes_json"]) == expected_hashes
        assert not state.attempt_rows()
    finally:
        state.close()


def test_run_config_pins_science_but_not_execution_timeout() -> None:
    config = _config()
    assert config["device"] == "cuda"
    assert config["dtype"] == "float64"
    assert config["lambda_grid"] is None
    assert config["lambda_grid_mode"] == "partition_guided_admm"
    assert config["selection_score"] == "partition_icl"
    assert config["evaluate_all_candidates"] is False
    assert config["inner_backend"] == "dense"
    assert config["workset_max_bytes"] == scheduler.DEFAULT_WORKSET_MAX_BYTES
    assert (
        config["compressed_cache_max_bytes"]
        == scheduler.DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    )
    assert config["dense_fallback_policy"] == "auto"
    assert config["allow_heuristic_structure_splits"] is True
    assert config["materialize_full_dual"] is False
    assert config["environment_sha256"] == "e" * 64
    assert "timeout_seconds" not in config
    assert "termination_grace_seconds" not in config
    changed_policy = _args()
    changed_policy.timeout_seconds = 9_999.0
    changed_policy.termination_grace_seconds = 99.0
    assert (
        scheduler.build_run_config(
            changed_policy,
            cohort_sha256="a" * 64,
            source_sha256="b" * 64,
            environment_sha256="e" * 64,
        )
        == config
    )

    quotient_args = _args()
    quotient_args.inner_backend = "quotient-workset"
    quotient_args.dense_fallback_policy = "cpu-allowed"
    quotient_args.workset_max_bytes = 12345
    quotient_args.materialize_full_dual = True
    quotient_config = scheduler.build_run_config(
        quotient_args,
        cohort_sha256="a" * 64,
        source_sha256="b" * 64,
        environment_sha256="e" * 64,
    )
    assert quotient_config["inner_backend"] == "quotient_workset"
    assert quotient_config["dense_fallback_policy"] == "cpu_allowed"
    options = scheduler._fit_options_from_config(quotient_config)
    assert options.inner_backend == "quotient_workset"
    assert options.dense_fallback_policy == "cpu_allowed"
    assert options.workset_max_bytes == 12345
    assert options.materialize_full_dual is True


def test_max_cases_selects_next_pending_cases(tmp_path: Path) -> None:
    inputs, truth = _cohort(tmp_path)
    manifest = scheduler.build_cohort_manifest(inputs, truth, expected_cases=2)
    state = scheduler.BenchmarkState(tmp_path / "state.sqlite3")
    try:
        state.initialize(manifest)
        first = scheduler.pending_scheduled_cases(state, manifest, max_cases=1)
        assert [case["tumor_id"] for case in first] == [CASE_SMALL]
        state.recover_success(
            CASE_SMALL,
            backend="cuda",
            bundle_path=tmp_path / "bundle",
            artifact_hashes={"artifact": "hash"},
        )
        second = scheduler.pending_scheduled_cases(state, manifest, max_cases=1)
        assert [case["tumor_id"] for case in second] == [CASE_LARGE]
    finally:
        state.close()


def test_scheduler_lock_refuses_a_second_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "scheduler.lock"
    with scheduler.SchedulerLock(lock_path):
        with pytest.raises(RuntimeError, match="Another benchmark scheduler"):
            with scheduler.SchedulerLock(lock_path):
                pass


def test_isolated_worker_timeout_terminates_process_group(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    returncode, timed_out, elapsed = scheduler.run_isolated_worker(
        command,
        log_path=tmp_path / "timeout.log",
        timeout_seconds=0.05,
        termination_grace_seconds=0.2,
    )
    assert timed_out is True
    assert returncode != 0
    assert elapsed < 5.0


def test_run_summary_provenance_is_added_atomically(tmp_path: Path) -> None:
    path = tmp_path / "summary.tsv"
    _write_tsv(path, ["tumor_id"], [[CASE_SMALL]])
    scheduler._add_run_summary_provenance(
        path, {"benchmark_config_sha256": CONFIG_SHA256}
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["benchmark_config_sha256"] == CONFIG_SHA256
