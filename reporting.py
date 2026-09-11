from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd

from ._version import __version__
from ._source import source_fingerprint, git_source_identity
from .config import _FitOptions, ALGORITHM_ID
from .core.fusion.types import RawFit
from .core.bic import effective_bic_mutation_region_count
from .core.objective import (
    compile_observed_model, make_base_objective_key, infer_integer_multiplicity_posterior_numpy,
)
from .io.data import CNFilterReport, TumorData, tumor_data_fingerprint, restore_immutable_record
from .model_selection.candidates import validate_candidate_identity
from .model_selection.proposals import pilot_matrix_hash
from .model_selection.types import (
    BICSelectionResult,
    RawFusionCandidate,
    SelectablePartitionCandidate,
    SelectionScore,
    DirectPartition,
    FusionPartition,
    PartitionRefitSummary,
)

SelectedPartition = FusionPartition | DirectPartition
OUTPUT_SUFFIXES = (
    "mutation_clusters.tsv", "cluster_centers.tsv",
    "mutation_region_multiplicity.tsv", "excluded_mutations.tsv",
    "run_manifest.json",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _source_identity() -> dict[str, object]:
    """Hash installed Python sources; never mistake an enclosing repo for ours."""
    root = Path(__file__).resolve().parent
    source_hash = source_fingerprint(root)
    commit, dirty = git_source_identity(root)
    if not (root / ".git").exists():
        # A wheel identity is valid only while its installed Python bytes
        # still match the package actually fingerprinted by the build hook.
        try:
            built = json.loads((root / "_build_source.json").read_text())
            if not isinstance(built, dict):
                raise ValueError("Build source identity must be a JSON object.")
            revision = built.get("commit")
            if (built.get("schema_version") == 1
                and built.get("python_source_sha256") == source_hash
                and isinstance(revision, str) and len(revision) == 40
                and all(character in "0123456789abcdef" for character in revision)
                and isinstance(built.get("dirty"), bool)):
                commit, dirty = revision, built["dirty"]
        except (OSError, ValueError, TypeError):
            pass
    return {"commit": commit, "dirty": dirty, "python_source_sha256": source_hash}


class RunPublication:
    """Exclusive tumor namespace with a completion-last, hash-bound manifest.

    A failed or interrupted attempt must use a new output directory to retry.
    Existing files are never replaced or removed, including partial attempts.
    """

    def __init__(
        self, outdir: Path, tumor_id: str, *, input_file: Path | None = None,
        expected_input_sha256: str | None = None,
        fit_config: _FitOptions | None = None, workflow: dict[str, object] | None = None,
    ) -> None:
        self.outdir, self.tumor_id = Path(outdir), str(tumor_id)
        if not self.tumor_id or Path(self.tumor_id).name != self.tumor_id or self.tumor_id in {".", ".."}:
            raise ValueError("tumor_id must be a nonempty filename component.")
        if input_file is not None:
            for suffix in OUTPUT_SUFFIXES:
                destination = self.outdir / f"{self.tumor_id}_{suffix}"
                if destination.resolve() == input_file.resolve() or (
                    destination.exists() and destination.samefile(input_file)
                ):
                    raise ValueError(f"Output would overwrite original input: {destination}. Choose a separate output directory.")
        input_sha256 = None if input_file is None else _file_hash(input_file)
        if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
            raise ValueError("Input changed while loading; no output was published.")
        self.outdir.mkdir(parents=True, exist_ok=True)
        conflicts = sorted(path.name for path in self.outdir.iterdir()
                           if path.name.startswith(f"{self.tumor_id}_"))
        if conflicts:
            raise FileExistsError(f"Existing tumor outputs must not be overwritten: {', '.join(conflicts)}. Choose a new output directory.")
        config = None
        if fit_config is not None:
            config = asdict(replace(fit_config, graph=replace(fit_config.graph, graph=None)))
            graph = fit_config.graph.graph
            config["graph"]["graph"] = None if graph is None else {"fingerprint": graph.fingerprint}
        workflow = {} if workflow is None else workflow
        self.path = self.outdir / f"{self.tumor_id}_run_manifest.json"
        self.record = {
            "schema_version": 1, "run_id": uuid4().hex, "tumor_id": self.tumor_id,
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "software_version": __version__, "source": _source_identity(),
            "algorithm_id": ALGORITHM_ID,
            "input": None if input_file is None else {
                "path": str(input_file.resolve()), "sha256": input_sha256,
            },
            "config": config, "config_sha256": None if config is None else hashlib.sha256(_json_bytes(config)).hexdigest(),
            "workflow": workflow, "workflow_sha256": hashlib.sha256(_json_bytes(workflow)).hexdigest(),
            "files": {}, "analysis": None,
        }
        # The manifest is also the exclusive claim: two concurrent starts
        # cannot both acquire the same namespace after their directory check.
        with self.path.open("xb") as stream:
            stream.write(_json_bytes(self.record))
            stream.flush()
            os.fsync(stream.fileno())

    def _save(self) -> None:
        current = json.loads(self.path.read_text())
        if current.get("run_id") != self.record["run_id"] or current.get("status") != "running":
            raise RuntimeError("Run manifest ownership or running status changed.")
        with tempfile.NamedTemporaryFile(dir=self.outdir, prefix=".clipp2-manifest-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(_json_bytes(self.record))
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _remember(self, path: Path) -> None:
        self.record["files"][path.name] = {"sha256": _file_hash(path), "size_bytes": path.stat().st_size}

    def write_audit(self, report: CNFilterReport | None) -> None:
        path = self.outdir / f"{self.tumor_id}_excluded_mutations.tsv"
        with path.open("x") as stream:
            cn_filter_output_table(self.tumor_id, report).to_csv(stream, sep="\t", index=False)
        self._remember(path)
        self._save()

    def fail(self, error: BaseException) -> None:
        self.record.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                           error={"type": type(error).__name__, "message": str(error)})
        try:
            self._save()
        except Exception as manifest_error:
            # A disk/full permission failure must not hide the fit exception.
            # The previous running manifest still cannot be read as complete.
            warnings.warn(f"Could not persist failed run status: {manifest_error}", RuntimeWarning)

    def publish(self, tables: dict[str, pd.DataFrame], *, analysis: dict[str, object] | None = None) -> None:
        if {f"{suffix}.tsv" for suffix in tables} != set(OUTPUT_SUFFIXES[:-1]):
            raise ValueError("Publication requires all four analysis tables.")
        # Qualification is supplied only after identity-valid table generation.
        # It describes the fit even if file publication subsequently fails.
        _json_bytes(analysis)
        self.record["analysis"] = analysis
        with tempfile.TemporaryDirectory(dir=self.outdir, prefix=".clipp2-tables-") as staging:
            paths = []
            for suffix, table in tables.items():
                path = Path(staging) / f"{self.tumor_id}_{suffix}.tsv"
                table.to_csv(path, sep="\t", index=False)
                paths.append(path)
            # Validate the early audit before publishing any fit table.
            for path in paths:
                existing = self.record["files"].get(path.name)
                if existing is not None and (
                    _file_hash(path) != existing["sha256"]
                    or _file_hash(self.outdir / path.name) != existing["sha256"]
                ):
                    raise ValueError("Early exclusion audit changed before publication.")
            for path in paths:
                if path.name not in self.record["files"]:
                    destination = self.outdir / path.name
                    os.link(path, destination)  # Atomic publication, never clobber.
                    self._remember(destination)
            for name, identity in self.record["files"].items():
                if _file_hash(self.outdir / name) != identity["sha256"]:
                    raise ValueError("Published output changed before completion.")
            expected = {self.path.name, *self.record["files"]}
            actual = {path.name for path in self.outdir.iterdir()
                      if path.name.startswith(f"{self.tumor_id}_")}
            if actual != expected:
                raise ValueError("Unexpected tumor outputs appeared during publication.")
            if self.record["input"] is not None:
                if _file_hash(Path(self.record["input"]["path"])) != self.record["input"]["sha256"]:
                    raise ValueError("Input changed during the run.")
            self.record.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
            self._save()


def _validated_profile(
    data: TumorData,
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float64)
    expected = (int(data.num_mutations), int(data.num_regions))
    if profile.shape != expected:
        raise ValueError(f"{name} has shape {profile.shape}; expected {expected}.")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} must contain only finite values.")
    return profile


def _mutation_output_table(analysis: AnalysisSerialization) -> pd.DataFrame:
    data, partition, refit = analysis.data, analysis.partition, analysis.refit
    labels, refit_phi = partition.labels, refit.phi
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "cluster_label": labels + 1,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = str(region_id)
        # The selected fixed-partition refit is the authoritative reported CCF.
        # Keep the compact v0.2.1-style public name requested by downstream
        # consumers; raw-fusion diagnostics remain in the audit tables.
        table[f"phi_{region}"] = refit_phi[:, column]
    return table


def _cluster_output_table(analysis: AnalysisSerialization) -> pd.DataFrame:
    data, partition, refit = analysis.data, analysis.partition, analysis.refit
    labels, centers = partition.labels, refit.cluster_centers
    sizes = np.bincount(labels, minlength=int(partition.n_clusters))
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, partition.n_clusters),
            "cluster_label": np.arange(1, partition.n_clusters + 1, dtype=int),
            "cluster_size": sizes,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = str(region_id)
        table[f"phi_{region}"] = centers[:, column]
    return table


def _add_integer_multiplicity(
    table: pd.DataFrame,
    *,
    data: TumorData,
    phi: np.ndarray,
    eps: float,
    max_major_cn: int,
) -> None:
    posterior = infer_integer_multiplicity_posterior_numpy(data, phi, eps=eps)
    count = posterior.candidate_count.reshape(-1)
    table["multiplicity_candidates"] = [
        ",".join(str(candidate) for candidate in range(1, int(size) + 1))
        for size in count
    ]
    table["multiplicity_candidate_count"] = count
    calls = pd.array(posterior.multiplicity_call.reshape(-1), dtype="Int64")
    informative = posterior.informative.reshape(-1)
    calls[(~informative) & (count > 1)] = pd.NA
    table["multiplicity_call"] = calls
    table["multiplicity_call_probability"] = posterior.map_probability.reshape(-1)
    table["multiplicity_informative"] = informative
    for candidate in range(1, max_major_cn + 1):
        table[f"multiplicity_p{candidate}"] = (
            posterior.posterior[..., candidate - 1].reshape(-1)
            if candidate <= posterior.posterior.shape[-1]
            else 0.0
        )


def _mutation_region_output_table(analysis: AnalysisSerialization) -> pd.DataFrame:
    data, partition, refit = analysis.data, analysis.partition, analysis.refit
    labels, refit_phi = partition.labels, refit.phi
    mutation_ids = np.repeat(
        np.asarray(data.mutation_ids, dtype=object), data.num_regions
    )
    region_ids = np.tile(
        np.asarray([str(x) for x in data.region_ids], dtype=object),
        data.num_mutations,
    )
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, mutation_ids.shape[0]),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "cluster_label": np.repeat(labels + 1, data.num_regions),
            "phi": refit_phi.reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
        }
    )
    _add_integer_multiplicity(table, data=data, phi=refit_phi,
                              eps=analysis.raw_fit.provenance.likelihood_eps,
                              max_major_cn=analysis.fit_config.max_major_cn)
    return table


def cn_filter_output_table(
    tumor_id: str,
    report: CNFilterReport | None,
) -> pd.DataFrame:
    """One row per triggering mutation-region-reason, also when none remain."""

    fields = (
        "mutation_id",
        "sample_id",
        "segment_id",
        "reason",
        "n_distinct_cn_states",
        "max_major_cn",
    )
    return pd.DataFrame(
        [
            {"tumor_id": tumor_id, **{name: getattr(record, name) for name in fields},
             "major_cn_limit": report.max_major_cn}
            for record in (() if report is None else report.records)
        ],
        columns=("tumor_id", *fields, "major_cn_limit"),
    )


def _write_fit_tables(analysis: AnalysisSerialization, publication: RunPublication) -> None:
    tables = {
        "mutation_clusters": _mutation_output_table(analysis),
        "cluster_centers": _cluster_output_table(analysis),
        "mutation_region_multiplicity": _mutation_region_output_table(analysis),
        "excluded_mutations": cn_filter_output_table(
            analysis.data.tumor_id, analysis.data.cn_filter_report
        ),
    }
    publication.publish(tables, analysis=analysis.qualification)


SUMMARY_SCHEMA_VERSION = 6


def input_model_summary(data: TumorData, *, eps=1e-6) -> dict[str, object]:
    """Separate eligibility provenance from the retained numerical model."""
    report = data.cn_filter_report
    records = () if report is None else report.records
    model = compile_observed_model(data, eps=eps)
    return {
        "input_mutation_count": data.num_mutations if report is None else report.input_mutation_count,
        "retained_mutation_count": data.num_mutations,
        "excluded_mutation_count": 0 if report is None else len(report.excluded_mutation_ids),
        "excluded_subclonal_cn_mutation_count": len({
            record.mutation_id for record in records if record.reason == "SUBCLONAL_CN_REGION"
        }),
        "excluded_major_cn_above_limit_mutation_count": len({
            record.mutation_id for record in records if record.reason == "MAJOR_CN_ABOVE_LIMIT"
        }),
        "max_major_cn": None if report is None else report.max_major_cn,
        "cn_filter_policy_id": None if report is None else report.policy_id,
        "multiplicity_model_id": model.model_id,
        "multiplicity_candidate_generator_version": model.candidate_generator_version,
        "multiplicity_prior_mode": model.prior_mode,
        "multiplicity_policy": model.support_policy,
        "multiplicity_coupling": model.coupling,
        "multiplicity_reduction_schema": 1,
    }


def _array_fingerprint(values: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _raw_qualification(fit: RawFit) -> dict[str, object]:
    """Record the authoritative, immutable raw result's own evidence."""
    certificate, provenance = fit.certificate, fit.provenance
    return {
        "kkt_certified": bool(certificate.certified),
        "admissible": bool(certificate.admissible),
        "global_optimum_certified": bool(certificate.global_optimum),
        "global_optimality_basis": str(provenance.global_optimality_basis),
        "certificate_status": str(certificate.status),
        "certificate_schema_version": int(certificate.schema_version),
        "kkt_residual": _finite_number(certificate.components.residual),
        "kkt_tolerance": _finite_number(certificate.tolerance),
        "solve_tolerance": _finite_number(fit.convergence.solve_tolerance),
        "residual_method": str(certificate.residual_method),
        "audit_dtype": str(certificate.audit_dtype),
        "working_dtype": str(certificate.working_dtype),
        "working_precision_kkt_residual": _finite_number(certificate.working_residual),
        "precision_polish_applied": bool(certificate.precision_polished),
        "precision_polish_max_abs_phi_delta": _finite_number(certificate.precision_polish_delta),
        "objective": _finite_number(fit.objective.total),
        "lambda": _finite_number(provenance.lambda_value),
        "objective_hash": str(provenance.certificate_problem_hash),
        "base_objective_hash": str(provenance.base_fusion_objective_hash),
        "graph_hash": str(provenance.original_graph_hash),
        "source_data_hash": str(provenance.source_data_hash),
        "phi_hash": _array_fingerprint(fit.phi, dtype=np.dtype(np.float64)),
    }


def _qualification(analysis: AnalysisSerialization) -> dict[str, object]:
    """Keep publication, raw admission, fixed-label refit, and search distinct."""
    raw_fit, partition, refit = analysis.raw_fit, analysis.partition, analysis.refit
    selection_result = analysis.selection_result
    selected = selection_result.selected_model.partition_candidate
    selected_raw = selected.raw_fit if isinstance(selected, RawFusionCandidate) else None
    selected_score = {**asdict(selected.score), "assignment_penalty": selected.score.assignment_penalty}
    selected_score = {key: _finite_number(value) if isinstance(value, float) else value
                      for key, value in selected_score.items()}
    direct = isinstance(partition, DirectPartition)
    return {
        "raw_reference": _raw_qualification(raw_fit),
        "selected_raw_fit": None if selected_raw is None else _raw_qualification(selected_raw),
        "selected_partition": {
            "family": "direct_partition" if direct else "raw_fusion",
            "source": str(partition.source), "signature": str(partition.signature),
            "labels_hash": _array_fingerprint(partition.labels, dtype=np.dtype(np.int64)),
            "n_clusters": int(partition.n_clusters),
            "raw_partition_certified": bool(not direct and partition.certified),
            "parent_raw_candidate_id": (int(partition.parent_raw_candidate_id)
                                        if direct and partition.parent_raw_candidate_id is not None else None),
            "parent_raw_lambda": (_finite_number(partition.parent_raw_lambda)
                                  if direct and partition.parent_raw_lambda is not None else None),
            "parent_raw_phi_hash": (partition.parent_raw_phi_hash or None) if direct else None,
        },
        "refit": {
            "multiplicity_policy": refit.multiplicity_policy,
            "local_stationarity_resolved": refit.locally_converged,
            "finite_candidate_found": bool(refit.finite_candidate_found),
            "numerically_resolved": bool(refit.refit_numerically_resolved),
            "global_optimum_certified": bool(refit.global_optimum_certified),
            "mode": str(refit.refit_mode),
            "certificate_method": str(refit.global_certificate_method),
            "global_lower_bound": _finite_number(refit.global_lower_bound),
            "global_optimality_gap": _finite_number(refit.global_optimality_gap),
            "loglik": _finite_number(refit.loglik),
            "phi_hash": _array_fingerprint(refit.phi, dtype=np.dtype(np.float64)),
            "centers_hash": _array_fingerprint(refit.cluster_centers, dtype=np.dtype(np.float64)),
            "source_data_hash": refit.source_data_hash,
            "likelihood_eps": refit.likelihood_eps,
        },
        "selection": {
            "status": ("resolved" if selection_result.selection_optimum_resolved else "provisional_unresolved"),
            "optimum_resolved": bool(selection_result.selection_optimum_resolved),
            "boundary_unresolved": bool(selection_result.selection_boundary_unresolved),
            "raw_lambda_path_resolved": bool(selection_result.raw_lambda_path_resolved),
            "global_hybrid_optimum_certified": bool(selection_result.global_hybrid_optimum_certified),
            "stop_reason": str(selection_result.adaptive_search_stop_reason),
            "method": str(selection_result.selection_method),
            "score": selected_score,
        },
    }


@dataclass(frozen=True, slots=True, init=False)
class AnalysisSerialization:
    """One immutable validation boundary shared by all reporting consumers."""

    data: TumorData
    input_file: Path
    fit_config: _FitOptions
    selection_result: BICSelectionResult
    raw_fit: RawFit = field(init=False)
    partition: SelectedPartition = field(init=False)
    refit: PartitionRefitSummary = field(init=False)
    _qualification_json: bytes = field(init=False, repr=False)

    def __init__(
        self, data: TumorData, input_file: Path,
        fit_config: _FitOptions, selection_result: BICSelectionResult,
    ) -> None:
        if not isinstance(selection_result, BICSelectionResult) or not isinstance(fit_config, _FitOptions):
            raise TypeError("Reporting requires a completed selection and resolved configuration.")
        if data.cn_filter_report is None or data.cn_filter_report.max_major_cn != fit_config.max_major_cn:
            raise ValueError("Reporting max_major_cn must match the original CN filtering report.")
        model = selection_result.selected_model
        raw_fit = model.raw_reference.raw_fit
        partition, refit = model.partition_candidate.partition, model.partition_candidate.refit
        if not isinstance(raw_fit, RawFit) or not isinstance(refit, PartitionRefitSummary):
            raise TypeError("Reporting requires typed RawFit and PartitionRefitSummary evidence.")
        if not isinstance(partition, (FusionPartition, DirectPartition)):
            raise TypeError("Reporting requires a typed selected partition.")
        epsilon = raw_fit.provenance.likelihood_eps
        if fit_config.eps != epsilon:
            raise ValueError("Reporting eps must match the fitted likelihood provenance.")
        source_hash = tumor_data_fingerprint(data)
        policy = raw_fit.provenance.multiplicity_policy
        if fit_config.multiplicity_policy != policy:
            raise ValueError("Reporting multiplicity policy differs from the fitted objective.")
        source_model = compile_observed_model(data, eps=epsilon)

        def bind(partition, refit, raw=None):
            if refit.multiplicity_policy != policy:
                raise ValueError("Refit and reporting multiplicity policies differ.")
            if tuple(partition.mutation_ids) != tuple(data.mutation_ids):
                raise ValueError("Reporting data do not match the fitted mutation ordering.")
            if refit.source_data_hash != source_hash or refit.likelihood_eps != epsilon:
                raise ValueError("Reporting data or epsilon do not match the fixed refit source identity.")
            _validated_profile(data, refit.phi, name="refit.phi")
            if isinstance(partition, FusionPartition) and not partition.certified:
                raise AssertionError("Refusing to serialize an uncertified raw partition.")
            if raw is not None:
                _validated_profile(data, raw.phi, name="raw_fit.phi")
                provenance = raw.provenance
                if provenance.source_data_hash != source_hash:
                    raise ValueError("Reporting data do not match the fitted source data identity.")
                expected = make_base_objective_key(source_model,
                    graph_hash=provenance.original_graph_hash, eps=epsilon)
                if provenance.likelihood_eps != epsilon or provenance.objective_key.base != expected:
                    raise ValueError("Reporting likelihood, box or epsilon identity does not match the fit.")
                if provenance.objective_key.base != raw_fit.provenance.objective_key.base:
                    raise ValueError("Reporting candidates do not share the frozen base objective.")

        candidates = (model.raw_reference, model.partition_candidate, model.partition_parent_raw)
        seen = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            validate_candidate_identity(candidate)
            bind(candidate.partition, candidate.refit,
                 candidate.raw_fit if isinstance(candidate, RawFusionCandidate) else None)
            score = candidate.score
            if (score.degrees_of_freedom != candidate.partition.n_clusters * data.num_regions
                or score.n_eff != effective_bic_mutation_region_count(data)):
                raise ValueError("Selection score dimensions do not match the reporting data.")
        if isinstance(partition, DirectPartition):
            parent = model.partition_parent_raw
            if partition.parent_raw_candidate_id is not None:
                parent_fit = None if parent is None else parent.raw_fit
                if (parent_fit is None
                    or partition.parent_raw_lambda != parent_fit.provenance.lambda_value
                    or partition.parent_raw_phi_hash != pilot_matrix_hash(parent_fit.phi)):
                    raise ValueError("Direct-partition parent-Phi provenance does not match the supplied raw parent.")
            elif (parent is not None or partition.parent_raw_phi_hash or partition.parent_raw_lambda is not None):
                raise ValueError("Direct-partition parent provenance is incomplete or spurious.")
        for name, value in (("data", data), ("input_file", input_file),
                            ("fit_config", fit_config), ("selection_result", selection_result),
                            ("raw_fit", raw_fit), ("partition", partition), ("refit", refit)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_qualification_json", _json_bytes(_qualification(self)))

    @property
    def qualification(self) -> dict[str, object]:
        return json.loads(self._qualification_json)

    def __reduce__(self):
        inputs = dict(data=self.data, input_file=self.input_file,
                      fit_config=self.fit_config, selection_result=self.selection_result)
        # Never serialize cached qualification as authority: copy/load must
        # rebind source data, numerical evidence and selected identity.
        return restore_immutable_record, (type(self), inputs)

    @property
    def selected_candidate(self) -> SelectablePartitionCandidate:
        return self.selection_result.selected_model.partition_candidate

    @property
    def raw_reference(self) -> RawFusionCandidate:
        return self.selection_result.selected_model.raw_reference

    @property
    def partition_parent_raw(self) -> RawFusionCandidate | None:
        return self.selection_result.selected_model.partition_parent_raw

    @property
    def score(self) -> SelectionScore:
        return self.selected_candidate.score


def analysis_summary(
    analysis: AnalysisSerialization,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Serialize schema-v5 diagnostics from the manifest qualification record."""

    if analysis.selection_result is None or analysis.fit_config is None:
        raise ValueError("An analysis summary requires the completed selection and configuration.")

    data = analysis.data
    fit_config = analysis.fit_config
    result = analysis.selection_result
    raw_fit = analysis.raw_fit
    parent_raw = analysis.partition_parent_raw
    selected_lambda = result.selected_lambda_representative
    qualification = analysis.qualification
    raw_reference_evidence = qualification["raw_reference"]
    partition_evidence = qualification["selected_partition"]
    refit_evidence = qualification["refit"]
    search_evidence = qualification["selection"]
    score_evidence = search_evidence["score"]
    scalar_pilots = raw_fit.provenance.scalar_pilot_certificates

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "tumor_id": data.tumor_id,
        "input_file": str(analysis.input_file),
        **input_model_summary(data, eps=raw_fit.provenance.likelihood_eps),
        "scalar_pilot_coordinate_count": len(scalar_pilots),
        "scalar_pilot_certified_coordinate_count": sum(
            item.globally_certified for item in scalar_pilots
        ),
        "scalar_pilot_all_coordinates_certified": (
            all(item.globally_certified for item in scalar_pilots)
            if scalar_pilots else None
        ),
        "scalar_pilot_attained_value": (
            sum(item.attained_value for item in scalar_pilots)
            if scalar_pilots else None
        ),
        "scalar_pilot_optimality_gap": (
            sum(item.optimality_gap for item in scalar_pilots)
            if scalar_pilots else None
        ),
        "computation_profile": "balanced",
        "selection_status": search_evidence["status"],
        "selection_contract_id": str(fit_config.selection.contract_id),
        "selection_optimum_resolved": search_evidence["optimum_resolved"],
        "selection_boundary_unresolved": search_evidence["boundary_unresolved"],
        "selection_hits_lower_boundary": bool(result.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(result.selection_hits_upper_boundary),
        "selected_lambda": (
            None if selected_lambda is None else float(selected_lambda)
        ),
        "raw_reference_lambda": raw_reference_evidence["lambda"],
        "raw_reference_objective_certified": bool(
            raw_reference_evidence["kkt_certified"] and raw_reference_evidence["admissible"]
        ),
        **{name: partition_evidence[key] for name, key in (
            ("selected_candidate_family", "family"),
            ("selected_partition_source", "source"),
            ("selected_partition_parent_lambda", "parent_raw_lambda"),
            ("selected_n_clusters", "n_clusters"),
            ("selected_partition_signature", "signature"),
            ("selected_partition_certified", "raw_partition_certified"),
            ("selected_labels_hash", "labels_hash"),
        )},
        "selected_partition_parent_phi_hash": partition_evidence["parent_raw_phi_hash"] or "",
        "selected_partition_parent_signature": (
            str(parent_raw.partition.signature) if parent_raw is not None else ""
        ),
        "raw_reference_phi_hash": raw_reference_evidence["phi_hash"],
        **{name: refit_evidence[key] for name, key in (
            ("selected_fixed_partition_refit_centers_hash", "centers_hash"),
            ("selected_refit_numerically_resolved", "numerically_resolved"),
            ("selected_refit_global_optimum_certified", "global_optimum_certified"),
            ("selected_refit_global_optimality_gap", "global_optimality_gap"),
            ("selected_refit_global_lower_bound", "global_lower_bound"),
            ("selected_refit_global_certificate_method", "certificate_method"),
        )},
        **{name: score_evidence[key] for name, key in (
            ("selection_score_name", "name"),
            ("selection_score", "value"),
            ("selection_score_numerical_uncertainty", "numerical_uncertainty"),
            ("selection_loglik", "loglik"),
            ("selection_df", "degrees_of_freedom"),
            ("selection_penalty", "penalty"),
            ("selection_n_eff", "n_eff"),
            ("selection_assignment_log_evidence", "assignment_log_evidence"),
            ("selection_assignment_code_weight", "assignment_code_weight"),
            ("selection_assignment_penalty", "assignment_penalty"),
            ("selection_assignment_dirichlet_alpha", "assignment_dirichlet_alpha"),
        )},
        **{
            f"{prefix}_{name}": None if evidence is None else evidence[key]
            for prefix, evidence in (("raw_reference", raw_reference_evidence),
                                     ("selected_raw", qualification["selected_raw_fit"]))
            for name, key in (("penalized_objective", "objective"),
                              ("kkt_residual", "kkt_residual"),
                              ("kkt_tolerance", "kkt_tolerance"),
                              ("solve_tolerance", "solve_tolerance"),
                              ("working_dtype", "working_dtype"),
                              ("working_precision_kkt_residual", "working_precision_kkt_residual"),
                              ("residual_method", "residual_method"),
                              ("audit_dtype", "audit_dtype"),
                              ("precision_polish_applied", "precision_polish_applied"),
                              ("precision_polish_max_abs_phi_delta", "precision_polish_max_abs_phi_delta"),
                              ("base_objective_hash", "base_objective_hash"),
                              ("graph_hash", "graph_hash"),
                              ("source_data_hash", "source_data_hash"))
        },
        "configured_raw_solver_primal_tol": float(fit_config.solver.tolerance),
        "selection_method": search_evidence["method"],
        "num_candidates": int(result.num_candidates),
        "num_candidates_certified": int(result.num_candidates_certified),
        "ward_candidate_pool_complete": bool(result.ward_candidate_pool_complete),
        "raw_lambda_path_complete": search_evidence["raw_lambda_path_resolved"],
        "global_hybrid_optimum_certified": search_evidence["global_hybrid_optimum_certified"],
        "search_stop_reason": search_evidence["stop_reason"],
        "device": str(raw_fit.provenance.device),
        "dtype": str(raw_fit.provenance.dtype),
        "elapsed_seconds": float(elapsed_seconds),
        "software_version": __version__,
    }


def write_analysis_outputs(
    analysis: AnalysisSerialization,
    *,
    outdir: Path,
    publication: RunPublication,
) -> None:
    """Write the retained-mutation tables and original-CN exclusion audit."""
    if publication.outdir.resolve() != Path(outdir).resolve() or publication.tumor_id != analysis.data.tumor_id:
        raise ValueError("Publication does not belong to this tumor and output directory.")
    _write_fit_tables(analysis, publication)


__all__ = [
    "AnalysisSerialization",
    "RunPublication",
    "SUMMARY_SCHEMA_VERSION",
    "analysis_summary",
    "write_analysis_outputs",
    "cn_filter_output_table",
]
