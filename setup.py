"""Embed honest source identity in wheels without importing the inference stack."""
import json
import hashlib
from pathlib import Path
from runpy import run_path

from setuptools import Extension, setup
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


root = Path(__file__).resolve().parent
native_sources = ["kernel/csrc/chain.cpp", "kernel/csrc/forest.cpp", "kernel/csrc/bindings.cpp"]
native_inputs = native_sources + ["kernel/csrc/chain.h"]
native_identity = hashlib.sha256(b"".join(
    name.encode() + b"\0" + (root / name).read_bytes() for name in native_inputs
)).hexdigest()
setup(cmdclass={"build_py": BuildPy}, ext_modules=[Extension(
    "CliPP2.kernel._native", sources=native_sources,
    depends=["kernel/csrc/chain.h"], language="c++",
    define_macros=[("CLIPP2_KERNEL_BUILD_ID", '"' + native_identity + '"')],
    extra_compile_args=["-O3", "-std=c++17"],
)])
