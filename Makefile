# Prefer this project's virtualenv, so `make` works without activating it.
# Falls back to python3, then python. Override with: make PYTHON=/path/to/python
PYTHON ?= $(shell \
	if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif command -v python3 >/dev/null 2>&1; then command -v python3; \
	else echo python; fi)

.PHONY: help install reproduce test lint annotate merge verify verify-new paper-tables demo clean

help:
	@echo "Using PYTHON = $(PYTHON)"
	@echo
	@echo "  install       Install dependencies into .venv (creating it if needed)"
	@echo "  reproduce     Re-run baselines, ablation and error analysis into results/"
	@echo "  test          Run the test suite"
	@echo "  lint          Run flake8 over the source and experiment packages"
	@echo "  annotate      Build the stratified annotation queue, then open the tool"
	@echo "  merge         Merge the annotated queue back into the evaluation set"
	@echo "                (use: make merge ARGS=--replace)"
	@echo "  verify        Score the human check of the model-assisted labels"
	@echo "  verify-new    Draw a fresh check sample, excluding the pilot documents"
	@echo "  paper-tables  Regenerate paper/tables/*.tex from results/"
	@echo "  demo          Re-render the README demo GIF from results/"
	@echo "  clean         Remove generated results and caches"

install:
	@test -d .venv || python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	@echo
	@echo 'Done. make targets use .venv automatically.'
	@echo 'For bare python commands, activate it first: source .venv/bin/activate'

reproduce:
	$(PYTHON) run.py --config config/config.yaml

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m flake8 src/ experiments/ tools/ main.py run.py quickstart.py \
		--select=F401,F841,W293,E302,E303 --max-line-length=120

annotate:
	$(PYTHON) -m experiments.build_annotation_queue --target 300
	@echo
	@echo "Opening the annotation tool. Load results/annotation_queue.csv in it."
	@command -v open >/dev/null 2>&1 && open tools/annotate.html || \
		echo "Open tools/annotate.html in your browser."

merge:
	$(PYTHON) -m experiments.merge_annotations $(ARGS)

verify:
	$(PYTHON) -m experiments.verify_labels

verify-new:
	$(PYTHON) -m experiments.verify_labels --fresh

paper-tables:
	$(PYTHON) -m experiments.export_latex

demo:
	$(PYTHON) tools/make_demo_gif.py

clean:
	rm -rf results/*.json results/*.csv results/tables/*.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
