# Prefer this project's virtualenv, so `make` works without activating it.
# Falls back to python3, then python. Override with: make PYTHON=/path/to/python
PYTHON ?= $(shell \
	if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif command -v python3 >/dev/null 2>&1; then command -v python3; \
	else echo python; fi)

.PHONY: help install reproduce test lint annotate merge verify verify-new verify-apply study paper-tables paper demo clean

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
	@echo "  verify-apply  Promote the verified answers to final labels"
	@echo "  study         Class-description studies: language, content, replication"
	@echo "  paper-tables  Regenerate paper/tables/*.tex from results/"
	@echo "  paper         Build paper/main.pdf (needs tectonic: brew install tectonic)"
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
	$(PYTHON) -m flake8 src/ experiments/ tools/ tests/ main.py run.py quickstart.py conftest.py

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

verify-apply:
	$(PYTHON) -m experiments.verify_labels --apply

study:
	$(PYTHON) -m experiments.run_description_study
	$(PYTHON) -m experiments.run_description_study --encoder paraphrase-multilingual-mpnet-base-v2
	$(PYTHON) -m experiments.run_language_match --no-seeds
	$(PYTHON) -m experiments.run_replication
	$(PYTHON) -m experiments.run_reuters
	$(PYTHON) -m experiments.lexical_gap
	$(PYTHON) -m experiments.run_gap_analysis
	$(PYTHON) -m experiments.run_predictor_search

paper-tables:
	$(PYTHON) -m experiments.export_latex

# The ACL style files are fetched at a pinned commit rather than vendored:
# the upstream repository carries no licence file.
ACL_STYLE_REV = d5adc823ff0f80f98c80405ca0ab66c68e684409
ACL_STYLE_URL = https://raw.githubusercontent.com/acl-org/acl-style-files/$(ACL_STYLE_REV)

paper/acl.sty paper/acl_natbib.bst:
	curl -sfL -o $@ $(ACL_STYLE_URL)/$(notdir $@)

paper: paper-tables paper/acl.sty paper/acl_natbib.bst
	cd paper && tectonic -X compile main.tex
	@echo "Wrote paper/main.pdf"

demo:
	$(PYTHON) tools/make_demo_gif.py

clean:
	rm -rf results/*.json results/*.csv results/tables/*.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
