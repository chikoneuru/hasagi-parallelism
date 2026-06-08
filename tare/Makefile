# HASAGI testbed convenience targets
#
# All targets auto-use .venv if it exists. Bootstrap with `make venv`.

VENV ?= .venv
PY   ?= $(VENV)/bin/python
PIP  ?= $(VENV)/bin/pip

.PHONY: venv install dev test lint smoke up down logs exp01 exp02 exp03 probe-nvml clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cpu -e .[dev]

install:
	$(PIP) install -e .

dev:
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cpu -e .[dev,rl,energy-api,docker]

test:
	$(PY) -m pytest -ra

lint:
	$(PY) -m ruff check hasagi tests experiments

smoke:
	$(PY) experiments/exp01_smoke_test.py

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f orchestrator

exp01:
	$(PY) experiments/exp01_smoke_test.py

exp02:
	$(PY) experiments/exp02_carbon_replay.py --trace traces/synthetic_solar.csv --hours 24

exp03:
	$(PY) experiments/exp03_elastic_reconfig.py

probe-nvml:
	$(PY) experiments/probe_nvml.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
