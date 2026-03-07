# Makefile for hjortMatrixWrapper
.PHONY: all clean build install test force

all: build

build:
	python3 pythonMakefile.py build_ext --inplace

clean:
	rm -rf build/ *.so *.pyc __pycache__/

force: clean build

install:
	pip install -e .

test: build
	python3 test.py

rebuild: clean build

verbose:
	python3 pythonMakefile.py build_ext --inplace --verbose

where:
	python3 -c "import hjortMatrixWrapper; print(hjortMatrixWrapper.__file__)"