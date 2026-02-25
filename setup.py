from setuptools import setup, find_packages

setup(
    name="hjortmath",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'hjortmath': ['*.so', '*.c'],  # Include C extensions
    },
    author="Erik Hjortsäter",
    description="A matrix library with C backend",
    python_requires=">=3.7",
)