from __future__ import annotations

import unittest

import numpy as np

from CliPP2.io.path_compiler import (
    LocalCopyNumberState,
    compile_single_switch_paths,
)


class PathPriorTests(unittest.TestCase):
    def test_high_copy_number_prior_stays_finite_in_log_space(self) -> None:
        compiled = compile_single_switch_paths(
            [LocalCopyNumberState(1.0, 551, 0)],
            allele_mode="unphased",
            dosage_prior_penalty=3.0,
        )
        log_prior = np.asarray(compiled.log_prior, dtype=np.float64)
        self.assertEqual(len(compiled.paths), 551)
        self.assertTrue(np.all(np.isfinite(log_prior)))
        self.assertAlmostEqual(float(np.logaddexp.reduce(log_prior)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
