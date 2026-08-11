from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from CliPP2.core.model import FitOptions
from CliPP2.model_selection.partitions import _partition_signature
from CliPP2.runners.pipeline import process_tumor_bundle


class PipelineIdentityTests(unittest.TestCase):
    def test_output_serialization_does_not_change_selection(self) -> None:
        input_file = Path(__file__).parent / "data" / "tinyTumor.tsv"
        options = FitOptions(lambda_value=0.0, device="cpu", dtype="float64")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "with-output"
            no_output_dir = root / "without-output"
            output_summary, output_search = process_tumor_bundle(
                input_file,
                output_dir,
                options,
                write_outputs=True,
            )
            no_output_summary, no_output_search = process_tumor_bundle(
                input_file,
                no_output_dir,
                options,
                write_outputs=False,
            )

            for key, value in output_summary.items():
                if key != "elapsed_seconds":
                    self.assertEqual(value, no_output_summary[key], key)
            self.assertFalse(no_output_dir.exists())

            elapsed_columns = [
                column
                for column in output_search.columns
                if "elapsed" in column or column.endswith("_seconds")
            ]
            pd.testing.assert_frame_equal(
                output_search.drop(columns=elapsed_columns),
                no_output_search.drop(columns=elapsed_columns),
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )

            selected_lambda = float(output_summary["selected_lambda"])
            selected = output_search.loc[
                output_search["bic_selection_eligible"].astype(bool)
                & np.isclose(
                    output_search["lambda"].to_numpy(dtype=float),
                    selected_lambda,
                    rtol=0.0,
                    atol=1e-12,
                )
            ].iloc[0]
            self.assertTrue(bool(selected["raw_objective_certified"]))
            self.assertTrue(bool(selected["raw_clonal_anchor_certified"]))
            self.assertTrue(bool(selected["raw_clonal_anchor_search_complete"]))
            self.assertEqual(
                int(selected["raw_clonal_anchor_candidates_evaluated"]), 4
            )
            self.assertEqual(
                int(selected["raw_clonal_anchor_frozen_coordinate_count"]), 2
            )
            self.assertGreaterEqual(int(selected["outer_num_frozen_coordinates"]), 2)
            self.assertTrue(bool(selected["partition_certified"]))
            self.assertTrue(bool(selected["partition_maximal"]))
            self.assertTrue(bool(selected["refit_numerically_resolved"]))
            self.assertFalse(bool(selected["refit_global_optimum_certified"]))
            self.assertEqual(output_search["objective_spec_hash"].nunique(), 1)
            self.assertEqual(output_search["original_graph_hash"].nunique(), 1)
            self.assertFalse(
                output_search["candidate_pool_source"]
                .astype(str)
                .str.contains("ward|cem", case=False, regex=True)
                .any()
            )
            self.assertFalse(
                any("reassignment" in column.lower() for column in output_search)
            )
            self.assertAlmostEqual(
                float(selected["selection_score"]),
                -2.0 * float(selected["selection_loglik"])
                + float(selected["selection_penalty"]),
                places=12,
            )

            mutation_table = pd.read_csv(
                output_dir / "tinyTumor_mutation_clusters.tsv",
                sep="\t",
            )
            self.assertEqual(
                _partition_signature(
                    mutation_table["selected_cluster_label"].to_numpy(dtype=np.int64)
                ),
                output_summary["selected_partition_signature"],
            )
            self.assertEqual(
                [
                    column
                    for column in mutation_table
                    if column == "selected_cluster_label"
                ],
                ["selected_cluster_label"],
            )
            anchor_rows = mutation_table.loc[
                mutation_table["raw_clonal_anchor_mutation"].astype(bool)
            ]
            self.assertEqual(len(anchor_rows), 1)
            anchor_row = anchor_rows.iloc[0]
            self.assertEqual(
                str(anchor_row["mutation_id"]),
                str(output_summary["selected_raw_clonal_anchor_mutation_id"]),
            )
            target = np.fromstring(
                str(output_summary["selected_raw_clonal_anchor_target"]), sep=","
            )
            raw_columns = [
                column for column in mutation_table if column.startswith("raw_phi_")
            ]
            np.testing.assert_allclose(
                anchor_row[raw_columns].to_numpy(dtype=np.float64),
                target,
                rtol=0.0,
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
