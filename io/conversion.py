from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SNV_CHROMOSOME_COL = "chromosome_index"
SNV_POSITION_COL = "position"
SNV_REF_COL = "ref_count"
SNV_ALT_COL = "alt_count"

CNA_CHROMOSOME_COL = "chromosome_index"
CNA_START_COL = "start_position"
CNA_END_COL = "end_position"
CNA_MAJOR_COL = "major_cn"
CNA_MINOR_COL = "minor_cn"

JEFFREYS_PSEUDO = 0.5


@dataclass(frozen=True)
class ConversionConfig:
    input_root: str | Path = "CliPP2Sim"
    output_root: str | Path = "CliPP2Sim_TSV"


def map_cna_to_mutations(mut_coords: pd.DataFrame, cna: pd.DataFrame) -> pd.DataFrame:
    mc = mut_coords.copy()
    mc[SNV_CHROMOSOME_COL] = mc[SNV_CHROMOSOME_COL].astype(str)

    cna = cna.copy()
    cna[CNA_CHROMOSOME_COL] = cna[CNA_CHROMOSOME_COL].astype(str)

    cross = mc.merge(
        cna,
        left_on=SNV_CHROMOSOME_COL,
        right_on=CNA_CHROMOSOME_COL,
        how="left",
    )
    in_segment = (
        (cross[SNV_POSITION_COL] >= cross[CNA_START_COL]) &
        (cross[SNV_POSITION_COL] <= cross[CNA_END_COL])
    )

    matches = cross.loc[
        in_segment,
        [SNV_CHROMOSOME_COL, SNV_POSITION_COL, CNA_MAJOR_COL, CNA_MINOR_COL],
    ].drop_duplicates()
    matches["has_cna"] = 1

    mapped = mc.merge(matches, on=[SNV_CHROMOSOME_COL, SNV_POSITION_COL], how="left")
    mapped["has_cna"] = mapped["has_cna"].fillna(0).astype(int)
    return mapped


def convert_one_patient(patient_path: Path, output_root: Path) -> Path | None:
    patient_id = patient_path.name
    sample_dirs = sorted(d for d in patient_path.iterdir() if d.is_dir())
    if not sample_dirs:
        return None

    mut_dfs = []
    for sample_dir in sample_dirs:
        snv_file = sample_dir / "snv.txt"
        if not snv_file.exists():
            continue
        snv = pd.read_csv(snv_file, sep="\t")
        mut_dfs.append(snv[[SNV_CHROMOSOME_COL, SNV_POSITION_COL]])

    if not mut_dfs:
        return None

    mut_catalog = pd.concat(mut_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    mut_catalog[SNV_CHROMOSOME_COL] = mut_catalog[SNV_CHROMOSOME_COL].astype(str)
    mut_catalog[SNV_POSITION_COL] = mut_catalog[SNV_POSITION_COL].astype(int)
    mut_catalog = mut_catalog.sort_values(by=[SNV_CHROMOSOME_COL, SNV_POSITION_COL]).reset_index(drop=True)
    mut_catalog["mutation_id"] = (
        mut_catalog[SNV_CHROMOSOME_COL].astype(str) + ":" + mut_catalog[SNV_POSITION_COL].astype(str)
    )

    per_sample_rows = []
    for sample_dir in sample_dirs:
        snv_file = sample_dir / "snv.txt"
        cna_file = sample_dir / "cna.txt"
        purity_file = sample_dir / "purity.txt"
        region_label = f"{patient_id}_{sample_dir.name.replace('sample', 'region')}"

        if not (snv_file.exists() and cna_file.exists() and purity_file.exists()):
            continue

        snv = pd.read_csv(snv_file, sep="\t")
        cna = pd.read_csv(cna_file, sep="\t")
        purity = float(purity_file.read_text(encoding="utf-8").strip())

        snv = snv[[SNV_CHROMOSOME_COL, SNV_POSITION_COL, SNV_REF_COL, SNV_ALT_COL]].copy()
        snv[SNV_CHROMOSOME_COL] = snv[SNV_CHROMOSOME_COL].astype(str)
        snv[SNV_POSITION_COL] = snv[SNV_POSITION_COL].astype(int)

        mut_sample = mut_catalog[[SNV_CHROMOSOME_COL, SNV_POSITION_COL, "mutation_id"]].copy()
        mut_sample = mut_sample.merge(snv, on=[SNV_CHROMOSOME_COL, SNV_POSITION_COL], how="left")
        cna_mapped = map_cna_to_mutations(mut_catalog[[SNV_CHROMOSOME_COL, SNV_POSITION_COL]], cna)
        mut_sample = mut_sample.merge(cna_mapped, on=[SNV_CHROMOSOME_COL, SNV_POSITION_COL], how="left")
        mut_sample.rename(
            columns={
                SNV_REF_COL: "ref_counts",
                SNV_ALT_COL: "alt_counts",
                CNA_MAJOR_COL: "major_cn",
                CNA_MINOR_COL: "minor_cn",
            },
            inplace=True,
        )

        mut_sample["region_id"] = region_label
        mut_sample["sample_id"] = region_label
        mut_sample["normal_cn"] = 2
        mut_sample["purity"] = purity

        per_sample_rows.append(
            mut_sample[
                [
                    SNV_CHROMOSOME_COL,
                    SNV_POSITION_COL,
                    "mutation_id",
                    "region_id",
                    "sample_id",
                    "ref_counts",
                    "alt_counts",
                    "normal_cn",
                    "major_cn",
                    "minor_cn",
                    "has_cna",
                    "purity",
                ]
            ].copy()
        )

    if not per_sample_rows:
        return None

    all_rows = pd.concat(per_sample_rows, ignore_index=True)
    for col in ["ref_counts", "alt_counts", "major_cn", "minor_cn", "has_cna"]:
        all_rows[col] = all_rows[col].astype(float)

    missing_counts = all_rows["ref_counts"].isna() | all_rows["alt_counts"].isna()
    all_rows.loc[missing_counts, "ref_counts"] = JEFFREYS_PSEUDO
    all_rows.loc[missing_counts, "alt_counts"] = JEFFREYS_PSEUDO

    missing_cna = all_rows["major_cn"].isna() | all_rows["minor_cn"].isna()
    all_rows.loc[missing_cna, "major_cn"] = 1.0
    all_rows.loc[missing_cna, "minor_cn"] = 1.0

    all_rows.sort_values(by=[SNV_CHROMOSOME_COL, SNV_POSITION_COL, "region_id"], inplace=True)
    all_rows.reset_index(drop=True, inplace=True)

    output_df = all_rows[
        [
            "mutation_id",
            "region_id",
            "sample_id",
            "ref_counts",
            "alt_counts",
            "normal_cn",
            "major_cn",
            "minor_cn",
            "has_cna",
            "purity",
        ]
    ].copy()

    output_root.mkdir(parents=True, exist_ok=True)
    out_file = output_root / f"{patient_id}.tsv"
    output_df.to_csv(out_file, sep="\t", index=False)
    return out_file


def convert_one_tumor(tumor_path: Path, output_root: Path) -> Path | None:
    return convert_one_patient(tumor_path, output_root)


def convert_simulation_root(
    input_root: str | Path = "CliPP2Sim",
    output_root: str | Path = "CliPP2Sim_TSV",
) -> list[Path]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
    written = []
    for patient_path in patient_dirs:
        out_file = convert_one_patient(patient_path, output_root)
        if out_file is not None:
            written.append(out_file)
            print(f"Wrote {out_file}")
    return written


def convert_simulation_root_from_config(config: ConversionConfig) -> list[Path]:
    return convert_simulation_root(input_root=config.input_root, output_root=config.output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge simulation folders into per-tumor TSV files.")
    parser.add_argument("--input-root", default="CliPP2Sim", help="Root directory containing tumor simulation folders.")
    parser.add_argument("--output-root", default="CliPP2Sim_TSV", help="Directory to write merged TSV files.")
    return parser


def conversion_config_from_args(args: argparse.Namespace) -> ConversionConfig:
    return ConversionConfig(input_root=args.input_root, output_root=args.output_root)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    written = convert_simulation_root_from_config(conversion_config_from_args(args))
    print(f"Converted {len(written)} tumors into {args.output_root}")


__all__ = [
    "ConversionConfig",
    "build_parser",
    "conversion_config_from_args",
    "convert_one_patient",
    "convert_one_tumor",
    "convert_simulation_root",
    "convert_simulation_root_from_config",
    "main",
    "map_cna_to_mutations",
]
