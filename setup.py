"""Embed honest source identity in wheels without importing the inference stack."""
import json
import os
from pathlib import Path
from runpy import run_path
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext


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


class BuildExt(build_ext):
    def build_extension(self, extension):
        if extension.name != "CliPP2.kernel._native_cuda":
            return super().build_extension(extension)
        compiler = shutil.which(os.environ.get("NVCC", "nvcc"))
        if compiler is None:
            raise RuntimeError("CLIPP2_BUILD_CUDA=1 requires nvcc; CPU builds do not need CUDA")
        output = Path(self.get_ext_fullpath(extension.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        architectures = os.environ.get("CLIPP2_CUDA_ARCHITECTURES", "80,89,90").split(",")
        if not architectures or any(not value.isdigit() for value in architectures):
            raise ValueError("CLIPP2_CUDA_ARCHITECTURES must contain comma-separated SM numbers")
        command = [compiler, "-std=c++17", "-O3", "--fmad=false", "-shared",
                   "-Xcompiler=-fPIC,-fno-fast-math,-ffp-contract=off",
                   '-DCLIPP2_BACKEND_BUILD_ID="' + backend_identity("cuda") + '"']
        host_compiler = os.environ.get("NVCC_CCBIN") or os.environ.get("CUDAHOSTCXX")
        if host_compiler:
            resolved = shutil.which(host_compiler)
            if resolved is None:
                raise RuntimeError("The explicitly selected CUDA host compiler is unavailable")
            command += ["--compiler-bindir", resolved]
        for architecture in architectures:
            command += [f"-gencode=arch=compute_{architecture},code=sm_{architecture}"]
        virtual = min(architectures, key=int)
        command += [f"-gencode=arch=compute_{virtual},code=compute_{virtual}",
                    *extension.sources, "-o", str(output)]
        subprocess.run(command, check=True)


root = Path(__file__).resolve().parent
build_helpers = run_path(str(root / "kernel/build_identity.py"))
backend_identity = build_helpers["backend_identity"]
backend_sources = build_helpers["backend_sources"]
native_inputs = list(build_helpers["NATIVE_SOURCES"])
extensions = [Extension(
    "CliPP2.kernel._native", sources=[name for name in native_inputs if name.endswith(".cpp")],
    depends=native_inputs, language="c++",
    define_macros=[("CLIPP2_KERNEL_BUILD_ID", '"' + build_helpers["native_identity"]() + '"')],
    extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-pthread"],
    extra_link_args=["-pthread"],
)]
openmp = ["-fopenmp"] if os.environ.get("CLIPP2_OPENMP") == "1" else []
extensions.append(Extension(
    "CliPP2.kernel._native_cpu", sources=["kernel/csrc/cpu/backend.cpp", "kernel/csrc/cpu/refit.cpp"],
    depends=[str(path.relative_to(root)) for path in backend_sources("cpu")], language="c++",
    define_macros=[("CLIPP2_BACKEND_BUILD_ID", '"' + backend_identity("cpu") + '"')],
    extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", *openmp],
    extra_link_args=openmp,
))
if os.environ.get("CLIPP2_BUILD_CUDA") == "1":
    extensions.append(Extension(
        "CliPP2.kernel._native_cuda", sources=["kernel/csrc/cuda/backend.cu"],
        depends=[str(path.relative_to(root)) for path in backend_sources("cuda")], language="c++"))
setup(cmdclass={"build_py": BuildPy, "build_ext": BuildExt}, ext_modules=extensions)
