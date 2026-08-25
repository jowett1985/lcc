UV ?= uv
VENV ?= .venv
APP ?= lcc
ENTRYPOINT ?= main.py

.PHONY: help venv sync format build clean

help:
	@printf '%s\n' \
		'make venv    Create the project virtual environment' \
		'make sync    Create/update the environment and install locked dependencies' \
		'make format  Format Python source with Ruff' \
		'make build   Build a standalone executable in dist/' \
		'make clean   Remove generated build artifacts'

venv:
	$(UV) venv $(VENV)

sync: venv
	$(UV) sync

format: sync
	$(UV) run --with ruff ruff format main.py agent_loop

build: sync
	$(UV) run --with pyinstaller pyinstaller \
		--clean \
		--onefile \
		--console \
		--name $(APP) \
		--paths . \
		$(ENTRYPOINT)

clean:
	rm -rf build dist *.spec
