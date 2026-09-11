"""Embed honest source identity in wheels without importing the inference stack."""
import json
from pathlib import Path
from runpy import run_path

from setuptools import setup
from setuptools.command.build_py import build_py


# Loading this stdlib-only file directly must not import CliPP2 or Torch.
source_helpers = run_path(str(Path(__file__).with_name("_source.py")))
source_fingerprint = source_helpers["source_fingerprint"]
git_source_identity = source_helpers["git_source_identity"]


class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [module for module in super().find_package_modules(package, package_dir)
                if module[1] != "setup"]

    def run(self):
        source = Path(__file__).resolve().parent
        commit, dirty = git_source_identity(source)
        super().run()
        package = Path(self.build_lib) / "CliPP2"
        record = {"schema_version": 1, "commit": commit, "dirty": dirty,
                  "python_source_sha256": source_fingerprint(package)}
        (package / "_build_source.json").write_text(json.dumps(record, sort_keys=True) + "\n")


setup(cmdclass={"build_py": BuildPy})
