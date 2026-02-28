from setuptools import setup, Extension

module = Extension(
    "hjortMatrixWrapper",
    sources=["hjortMatrixWrapper.c"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp", "-lgomp"],
)

setup(
    name="hjortMatrixWrapper",
    version="1.0",
    ext_modules=[module]
)