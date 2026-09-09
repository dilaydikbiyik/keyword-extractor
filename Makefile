PYTHON ?= python

.PHONY: help install reproduce test lint annotate paper-tables demo clean

help:
	@echo "install    Install dependencies into the active environment"
	@echo "reproduce  Re-run baselines, ablation and error analysis into results/"
	@echo "test       Run the test suite"
	@echo "lint       Run flake8 over the source and experiment packages"
	@echo "annotate   Build the stratified annotation queue for new labels"
	@echo "paper-tables  Regenerate paper/tables/*.tex from results/"
	@echo "demo       Re-render the README demo GIF from results/"
	@echo "clean      Remove generated results and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

reproduce:
	$(PYTHON) run.py --config config/config.yaml

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m flake8 src/ experiments/ tools/ main.py run.py quickstart.py \
		--select=F401,F841,W293,E302,E303 --max-line-length=120

annotate:
	$(PYTHON) -m experiments.build_annotation_queue --target 300

paper-tables:
	$(PYTHON) -m experiments.export_latex

demo:
	$(PYTHON) tools/make_demo_gif.py

clean:
	rm -rf results/*.json results/*.csv results/tables/*.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
