NANOBIND_DIR ?= $(error Set NANOBIND_DIR to the nanobind source directory)

VENV = .venv
PYTHON = $(VENV)/bin/python

CMAKE_ARGS = -DNANOBIND_DIR=$(NANOBIND_DIR) -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR=$(VENV)

ifdef XTENSOR_DIR
	CMAKE_ARGS += -Dxtensor_DIR=$(XTENSOR_DIR)
endif
ifdef XTL_DIR
	CMAKE_ARGS += -Dxtl_DIR=$(XTL_DIR)
endif
ifdef XSIMD_DIR
	CMAKE_ARGS += -Dxsimd_DIR=$(XSIMD_DIR)
endif
ifdef XTENSOR_PYTHON_DIR
	CMAKE_ARGS += -DXTENSOR_PYTHON_DIR=$(XTENSOR_PYTHON_DIR)
	CMAKE_ARGS += -Dpybind11_DIR=$(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
endif
ifdef CMAKE_CXX_COMPILER
	CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(CMAKE_CXX_COMPILER)
endif

.PHONY: venv build benchmark benchmark-nanobind benchmark-xtensor-python clean

venv:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install numpy
ifdef XTENSOR_PYTHON_DIR
	$(PYTHON) -m pip install pybind11
endif

build: venv
	cmake -S . -B build $(CMAKE_ARGS)
	cmake --build build -j

benchmark:
	PYTHONPATH=build $(PYTHON) scripts/run_benchmarks.py

benchmark-nanobind:
	PYTHONPATH=build $(PYTHON) scripts/run_benchmarks.py --backend nanobind

benchmark-xtensor-python:
	PYTHONPATH=build $(PYTHON) scripts/run_benchmarks.py --backend xtensor-python

clean:
	rm -rf build .venv
