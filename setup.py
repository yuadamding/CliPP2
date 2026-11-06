"""Build native extensions without importing the inference stack."""
import os
from pathlib import Path
from runpy import run_path
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


root = Path(__file__).resolve().parent
package_source = Path("src/CliPP2")


class BuildExt(build_ext):
    def build_extension(self, extension):
        if extension.name != "CliPP2.kernel._native_cuda":
            return super().build_extension(extension)
        compiler = shutil.which(os.environ.get("NVCC", "nvcc"))
        if compiler is None:
            raise RuntimeError("CLIPP2_BUILD_CUDA=1 requires nvcc; CPU builds do not need CUDA")
        output = Path(self.get_ext_fullpath(extension.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        architectures = os.environ.get("CLIPP2_CUDA_ARCHITECTURES", "80,86,89,90").split(",")
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


build_helpers = run_path(str(root / package_source / "kernel/build_identity.py"))
backend_identity = build_helpers["backend_identity"]
backend_sources = build_helpers["backend_sources"]
native_inputs = [str(package_source / name) for name in build_helpers["NATIVE_SOURCES"]]
extensions = [Extension(
    "CliPP2.kernel._native", sources=[name for name in native_inputs if name.endswith(".cpp")],
    depends=native_inputs, language="c++",
    define_macros=[("CLIPP2_KERNEL_BUILD_ID", '"' + build_helpers["native_identity"]() + '"')],
    extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-pthread"],
    extra_link_args=["-pthread"],
)]
openmp = ["-fopenmp"] if os.environ.get("CLIPP2_OPENMP") == "1" else []
extensions.append(Extension(
    "CliPP2.kernel._native_cpu", sources=[str(package_source / "kernel/csrc/cpu" / name)
                                        for name in ("backend.cpp", "refit.cpp")],
    depends=[str(path.relative_to(root)) for path in backend_sources("cpu")], language="c++",
    define_macros=[("CLIPP2_BACKEND_BUILD_ID", '"' + backend_identity("cpu") + '"')],
    extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", *openmp],
    extra_link_args=openmp,
))
if os.environ.get("CLIPP2_BUILD_CUDA") == "1":
    extensions.append(Extension(
        "CliPP2.kernel._native_cuda", sources=[str(package_source / "kernel/csrc/cuda/backend.cu")],
        depends=[str(path.relative_to(root)) for path in backend_sources("cuda")], language="c++"))
setup(cmdclass={"build_ext": BuildExt}, ext_modules=extensions)
