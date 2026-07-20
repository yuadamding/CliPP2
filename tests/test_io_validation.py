"""Input-domain validation tests (audit §5.3).

Verifies that load_tumor_tsv in strict mode rejects every category of
invalid input with a ValueError, and that a valid TSV loads cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CliPP2.io.data import load_tumor_tsv


_HEADER = (
    "mutation_id\tsample_id\tref_counts\talt_counts"
    "\tmajor_cn\tminor_cn\tpurity\tnormal_cn\thas_cna"
)


def _row(
    mutation_id: str = "mut1",
    sample_id: str = "s1",
    ref_counts: float = 10,
    alt_counts: float = 5,
    major_cn: float = 2,
    minor_cn: float = 1,
    purity: float = 0.8,
    normal_cn: float = 2,
    has_cna: int = 1,
) -> str:
    return (
        f"{mutation_id}\t{sample_id}\t{ref_counts}\t{alt_counts}"
        f"\t{major_cn}\t{minor_cn}\t{purity}\t{normal_cn}\t{has_cna}"
    )


def _tsv(*rows: str) -> str:
    return "\n".join([_HEADER] + list(rows)) + "\n"


def _write(tmp_path: Path, content: str, name: str = "tumor.tsv") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ── Baseline: a valid TSV must load without error ────────────────────────────


def test_valid_tsv_loads_without_error(tmp_path: Path) -> None:
    p = _write(tmp_path, _tsv(_row("mut1"), _row("mut2")))
    data = load_tumor_tsv(p)
    assert data.num_mutations == 2
    assert data.num_regions == 1
    assert data.alt_counts.dtype == np.float64
    assert data.total_counts.dtype == np.float64


# ── Negative / fractional counts ─────────────────────────────────────────────


def test_strict_rejects_negative_alt_counts(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", alt_counts=-1, ref_counts=10), _row("mut2"))
    with pytest.raises(ValueError, match="alt_counts"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_negative_total_counts(tmp_path: Path) -> None:
    # alt=0, ref=-10 → total = -10
    content = _tsv(_row("mut1", alt_counts=0, ref_counts=-10), _row("mut2"))
    with pytest.raises(ValueError, match="total_counts"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_alt_exceeds_total(tmp_path: Path) -> None:
    # alt=6, ref=-1 → total=5; alt=6 > total+0.5=5.5 → error
    content = _tsv(_row("mut1", alt_counts=6, ref_counts=-1), _row("mut2"))
    with pytest.raises(ValueError):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_fractional_alt_counts(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", alt_counts=4.7, ref_counts=5), _row("mut2"))
    with pytest.raises(ValueError, match="Non-integer"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_fractional_ref_counts(tmp_path: Path) -> None:
    # total = alt + ref = 4 + 5.6 = 9.6 → "Non-integer total_counts"
    content = _tsv(_row("mut1", alt_counts=4, ref_counts=5.6), _row("mut2"))
    with pytest.raises(ValueError, match="Non-integer"):
        load_tumor_tsv(_write(tmp_path, content))


# ── Purity ────────────────────────────────────────────────────────────────────


def test_strict_rejects_zero_purity(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", purity=0.0), _row("mut2", purity=0.0))
    with pytest.raises(ValueError, match="[Pp]urity"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_negative_purity(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", purity=-0.2), _row("mut2", purity=-0.2))
    with pytest.raises(ValueError, match="[Pp]urity"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_purity_above_one(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", purity=1.5), _row("mut2", purity=1.5))
    with pytest.raises(ValueError, match="[Pp]urity"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_accepts_purity_equal_to_one(tmp_path: Path) -> None:
    # purity=1.0 is valid; earlier code silently clipped to 1-eps
    content = _tsv(_row("mut1", purity=1.0), _row("mut2", purity=1.0))
    data = load_tumor_tsv(_write(tmp_path, content))
    assert np.all(data.purity == 1.0)


# ── Copy-number ordering ──────────────────────────────────────────────────────


def test_strict_rejects_major_less_than_minor(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", major_cn=1, minor_cn=2), _row("mut2"))
    with pytest.raises(ValueError, match="major"):
        load_tumor_tsv(_write(tmp_path, content))


# ── Normal CN ────────────────────────────────────────────────────────────────


def test_strict_rejects_zero_normal_cn(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", normal_cn=0), _row("mut2", normal_cn=0))
    with pytest.raises(ValueError, match="normal_cn"):
        load_tumor_tsv(_write(tmp_path, content))


def test_strict_rejects_negative_normal_cn(tmp_path: Path) -> None:
    content = _tsv(_row("mut1", normal_cn=-1), _row("mut2", normal_cn=-1))
    with pytest.raises(ValueError, match="normal_cn"):
        load_tumor_tsv(_write(tmp_path, content))


# ── Lenient mode ─────────────────────────────────────────────────────────────


def test_lenient_mode_skips_domain_validation(tmp_path: Path) -> None:
    # Negative alt_counts would raise in strict mode; lenient allows it through
    content = _tsv(_row("mut1", alt_counts=-1, ref_counts=10), _row("mut2"))
    # lenient mode should not raise a domain-validation ValueError;
    # it may still raise for non-finite values or other structural issues
    try:
        load_tumor_tsv(_write(tmp_path, content), validation_mode="lenient")
    except ValueError as exc:
        pytest.fail(f"lenient mode raised ValueError: {exc}")


# ── Structural checks (always active regardless of validation_mode) ───────────


def test_duplicate_mutation_region_pairs_are_always_rejected(tmp_path: Path) -> None:
    content = _tsv(
        _row("mut1", sample_id="s1"),
        _row("mut1", sample_id="s1"),  # duplicate
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_tumor_tsv(_write(tmp_path, content))


def test_missing_required_column_is_always_rejected(tmp_path: Path) -> None:
    # No purity column → load_tumor_tsv raises
    broken = (
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\thas_cna\n"
        "mut1\ts1\t10\t5\t2\t1\t1\n"
        "mut2\ts1\t8\t3\t2\t1\t1\n"
    )
    with pytest.raises(ValueError):
        load_tumor_tsv(_write(tmp_path, broken))
