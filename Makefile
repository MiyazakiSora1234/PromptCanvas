# PromptCanvas tasks. Works with GNU Make on Windows (cmd.exe) and macOS/Linux.
# Run `make help` for the list of targets.

BACKEND  := backend
FRONTEND := frontend
HOST     ?= 127.0.0.1
PORT     ?= 8000
# PyTorch build: cu128 (NVIDIA, incl. RTX 50 series), cu126, cpu, ...
TORCH    ?= cu128

# Quoted forward-slash paths work in both cmd.exe and sh (Git Bash), whichever make picks.
ifeq ($(OS),Windows_NT)
  PYTHON  ?= python
  PY      := ".venv/Scripts/python.exe"
else
  PYTHON  ?= python3
  PY      := ".venv/bin/python"
endif

.DEFAULT_GOAL := help
.PHONY: help setup venv install-torch install install-dev frontend-install env download-model \
        build run dev-api dev-web test test-backend test-frontend lint lint-backend lint-frontend \
        format typecheck typecheck-backend typecheck-frontend check clean

help: ## Show this help
	@echo Setup:
	@echo   setup             venv + PyTorch(TORCH=$(TORCH)) + Python deps + npm deps + frontend build
	@echo   install-torch     Install PyTorch build TORCH=cu128 / cpu / ...
	@echo   env               Create backend/.env from .env.example if missing
	@echo   download-model    Pre-download the configured model into the HF cache
	@echo Run:
	@echo   run               Build the frontend and serve everything on http://$(HOST):$(PORT)
	@echo   dev-api           API only (use together with dev-web)
	@echo   dev-web           Vite dev server with hot reload on http://localhost:5173 (proxies /api)
	@echo Quality:
	@echo   check             lint + typecheck + test for backend and frontend
	@echo   test / lint / typecheck   Both sides; or append -backend / -frontend
	@echo   format            Auto-fix and format Python with ruff
	@echo   clean             Remove venv, node_modules, build output and tool caches

# --- Setup -------------------------------------------------------------------

setup: venv install-torch install install-dev frontend-install build

venv:
	cd $(BACKEND) && $(PYTHON) -m venv .venv
	cd $(BACKEND) && $(PY) -m pip install --upgrade pip

install-torch:
	cd $(BACKEND) && $(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$(TORCH)
	cd $(BACKEND) && $(PY) -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

install:
	cd $(BACKEND) && $(PY) -m pip install -r requirements.txt

install-dev:
	cd $(BACKEND) && $(PY) -m pip install -r requirements-dev.txt

frontend-install:
	cd $(FRONTEND) && npm ci

env:
	cd $(BACKEND) && $(PY) -c "import pathlib, shutil; p = pathlib.Path('.env'); print('.env already exists') if p.exists() else (shutil.copy('.env.example', p), print('created backend/.env'))"

download-model:
	cd $(BACKEND) && $(PY) -m scripts.download_model

# --- Run ---------------------------------------------------------------------

build:
	cd $(FRONTEND) && npm run build

run: build
	cd $(BACKEND) && $(PY) -m uvicorn app.main:create_app --factory --host $(HOST) --port $(PORT)

dev-api:
	cd $(BACKEND) && $(PY) -m uvicorn app.main:create_app --factory --host $(HOST) --port $(PORT)

dev-web:
	cd $(FRONTEND) && npm run dev

# --- Quality -----------------------------------------------------------------

check: lint typecheck test

test: test-backend test-frontend
test-backend:
	cd $(BACKEND) && $(PY) -m pytest
test-frontend:
	cd $(FRONTEND) && npm test

lint: lint-backend lint-frontend
lint-backend:
	cd $(BACKEND) && $(PY) -m ruff check .
lint-frontend:
	cd $(FRONTEND) && npm run lint

typecheck: typecheck-backend typecheck-frontend
typecheck-backend:
	cd $(BACKEND) && $(PY) -m mypy
typecheck-frontend:
	cd $(FRONTEND) && npm run typecheck

format:
	cd $(BACKEND) && $(PY) -m ruff check --fix .
	cd $(BACKEND) && $(PY) -m ruff format .

clean:
	$(PYTHON) -c "import pathlib, shutil; b = pathlib.Path('$(BACKEND)'); f = pathlib.Path('$(FRONTEND)'); dirs = [b / d for d in ('.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache')] + [f / 'node_modules', f / 'dist'] + [p for p in b.rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(d, ignore_errors=True) for d in dirs]; print('cleaned')"
