from __future__ import annotations

from pathlib import Path

from CliPP2.core.fusion.types import ExactSolverResourceLimit
from CliPP2.runners import pipeline


def test_directory_run_records_exact_resource_limit_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    (input_dir / "a_limited.tsv").touch()
    (input_dir / "b_ok.tsv").touch()

    def fake_process_one_file(*, file_path, **_kwargs):
        path = Path(file_path)
        if path.stem == "a_limited":
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: forced by regression test"
            )
        return {
            "tumor_id": path.stem,
            "selection_eligible": True,
            "failure_reason": "converged",
        }

    monkeypatch.setattr(pipeline, "process_one_file", fake_process_one_file)

    result = pipeline.run_directory(
        input_dir,
        output_dir,
        workers=1,
        write_outputs=False,
    )

    assert result["tumor_id"].tolist() == ["a_limited", "b_ok"]
    limited = result.loc[result["tumor_id"] == "a_limited"].iloc[0]
    assert not bool(limited["selection_eligible"])
    assert not bool(limited["full_kkt_certified"])
    assert limited["full_kkt_certificate_status"] == "resource_limit"
    assert limited["failure_reason"] == "exact_solver_resource_limit"
    assert "forced by regression test" in limited["error_message"]
    ok = result.loc[result["tumor_id"] == "b_ok"].iloc[0]
    assert bool(ok["selection_eligible"])
    assert (output_dir / "single_stage_summary.tsv").exists()
