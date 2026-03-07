from setuptools import setup, Extension
import shutil
import os

if os.path.exists("build"):
    shutil.rmtree("build")

module = Extension(
    "hjortMatrixWrapper",
    sources=["hjortMatrixWrapper.c"],
    libraries=[
        "lapacke",
        "lapack",
        "openblas",
    ],
    library_dirs=[],
    include_dirs=[],
    extra_compile_args=[
        "-Wall",
        "-O3",
        "-fopenmp",
        "-mavx2",
        "-march=x86-64-v3",
        "-D_GNU_SOURCE",
    ],
    extra_link_args=["-fopenmp"],
)

setup(
    name="hjortMatrixWrapper",
    version="1.0",
    ext_modules=[module],
)

print("\n" + "="*50)
print("BUILD COMPLETE - Module location:")
print(os.path.abspath("hjortMatrixWrapper.cpython-312-x86_64-linux-gnu.so"))
print("="*50 + "\n")