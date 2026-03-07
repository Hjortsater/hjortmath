from setuptools import setup, Extension

common_args = {
    "extra_compile_args": ["-Wall", "-O3", "-fopenmp", "-mavx2", "-march=x86-64-v3"],
    "extra_link_args": ["-fopenmp"],
    "libraries": ["lapacke", "lapack", "openblas"]
}



# Extension 1: Standard Matrix
ext1 = Extension(
    "hjortMatrixWrapper",
    sources=["hjortMatrixWrapper.c", "hjortMatrixBackend.c"],
    **common_args
)

# Extension 2: Lazy Evaluation
# ADD 'hjortLazyEvaluate.c' HERE
ext2 = Extension(
    "hjortLazyMatrixWrapper",
    sources=["hjortLazyMatrixWrapper.c", "hjortMatrixBackend.c", "hjortLazyEvaluate.c"],
    **common_args
)

setup(
    name="hjortMath",
    version="1.0",
    ext_modules=[ext1, ext2],
)