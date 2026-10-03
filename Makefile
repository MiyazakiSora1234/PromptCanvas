# PromptCanvas tasks. Works with GNU Make on Windows (cmd.exe) and macOS/Linux.
# Run `make help` for the list of targets.

BACKEND := backend
HOST    ?= 127.0.0.1
PORT    ?= 8000
# PyTorch build: cu128 (NVIDIA, incl. RTX 50 series), cu126, cpu, ...
TORCH   ?= cu128

# Quoted forward-slash paths work in both cmd.exe and sh (Git Bash), whichever make picks.
ifeq ($(OS),Windows_NT)
  PYTHON  ?= python
  PY      := ".venv/Scripts/python.exe"
else
  PYTHON  ?= python3
  PY      := ".venv/bin/python"
endif

.DEFAULT_GOAL := help
.PHONY: help setup venv install-torch install install-dev env download-model run test lint format typecheck check clean

help: ## Show this help
	@echo Targets:
	@echo   setup           venv + PyTorch(TORCH=$(TORCH)) + app and dev dependencies
	@echo   venv            Create backend/.venv
	@echo   install-torch   Install PyTorch build TORCH=cu128 / cpu / ...
	@echo   install         Install app dependencies (requirements.txt)
	@echo   install-dev     Install test and lint tools (requirements-dev.txt)
	@echo   env             Create backend/.env from .env.example if missing
	@echo   download-model  Pre-download the configured model into the HF cache
	@echo   run             Start the server on http://$(HOST):$(PORT)
	@echo   test            Run pytest
	@echo   lint            Run ruff
	@echo   format          Auto-fix and format with ruff
	@echo   typecheck       Run mypy
	@echo   check           lint + typecheck + test
	@echo   clean           Remove venv and tool caches

setup: venv install-torch install install-dev ## Full setup

venv:
	cd $(BACKEND) && $(PYTHON) -m venv .venv
	cd $(BACKEND) && $(PY) -m pip install --upgrade pip

install-torch:
	cd $(BACKEND) && $(PY) -m pip install torch --index-url https://download.pytorch.org/whl/$(TORCH)
	cd $(BACKEND) && $(PY) -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

install:
	cd $(BACKEND) && $(PY) -m pip install -r requirements.txt

install-dev:
	cd $(BACKEND) && $(PY) -m pip install -r requirements-dev.txt

env:
	cd $(BACKEND) && $(PY) -c "import pathlib, shutil; p = pathlib.Path('.env'); print('.env already exists') if p.exists() else (shutil.copy('.env.example', p), print('created backend/.env'))"

download-model:
	cd $(BACKEND) && $(PY) -m scripts.download_model

run:
	cd $(BACKEND) && $(PY) -m uvicorn app.main:create_app --factory --host $(HOST) --port $(PORT)

test:
	cd $(BACKEND) && $(PY) -m pytest

lint:
	cd $(BACKEND) && $(PY) -m ruff check .

format:
	cd $(BACKEND) && $(PY) -m ruff check --fix .
	cd $(BACKEND) && $(PY) -m ruff format .

typecheck:
	cd $(BACKEND) && $(PY) -m mypy

check: lint typecheck test

clean:
	cd $(BACKEND) && $(PYTHON) -c "import pathlib, shutil; dirs = [pathlib.Path(d) for d in ('.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache')] + [p for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(d, ignore_errors=True) for d in dirs]; print('cleaned')"
