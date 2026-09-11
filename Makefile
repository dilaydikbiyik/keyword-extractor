# Prefer this project's virtualenv, so `make` works without activating it.
# Falls back to python3, then python. Override with: make PYTHON=/path/to/python
PYTHON ?= $(shell \
	if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif command -v python3 >/dev/null 2>&1; then command -v python3; \
	else echo python; fi)

.PHONY: help install reproduce test lint annotate merge verify verify-new verify-apply second-annotator second-annotator-score llm-baseline llm-baseline-local study rocchio-preregister rocchio rocchio-definitions-preregister rocchio-definitions paper-tables paper submission demo clean

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
	@echo "  second-annotator        Blind German-only sample for a second annotator"
	@echo "  second-annotator-score  Human-versus-human agreement once it is filled in"
	@echo "  llm-baseline  Run the paid LLM zero-shot baseline (needs OPENAI_API_KEY)"
	@echo "  llm-baseline-local  Same prompt, open model run locally (downloads 15.2 GB once)"
	@echo "  study         Class-description studies: language, content, replication"
	@echo "  rocchio-preregister  Label-free causal test, step 1: write the prediction"
	@echo "  rocchio       Step 2: accuracy, checked against the committed prediction"
	@echo "  rocchio-definitions-preregister / rocchio-definitions  The same, from written definitions"
	@echo "  paper-tables  Regenerate paper/tables/*.tex from results/"
	@echo "  paper         Build paper/main.pdf (needs tectonic: brew install tectonic)"
	@echo "  submission    Anonymous review PDF and code archive in dist/, checked"
	@echo "  demo          Re-render the README demo GIF from results/"
	@echo "  clean         Remove generated results and caches"

install:
	@test -d .venv || python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	@echo
	@echo 'Done. make targets use .venv automatically.'
	@echo 'For bare python commands, activate it first: source .venv/bin/activate'

# --extra-ablations: the mpnet and English-pivot rows are in the paper, so a
# reproduction that skips them does not reproduce the paper.
reproduce:
	$(PYTHON) run.py --config config/config.yaml --extra-ablations

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

second-annotator:
	$(PYTHON) -m experiments.verify_labels --second-build

second-annotator-score:
	$(PYTHON) -m experiments.verify_labels --second-score

# Billed to the account behind OPENAI_API_KEY. gpt-4o-mini is a different model
# family from the labeller, which a fair comparison against these labels needs.
llm-baseline:
	@test -n "$$OPENAI_API_KEY" || { echo "Set OPENAI_API_KEY first; this run is billed to that account."; exit 1; }
	$(PYTHON) -m experiments.run_experiments --with-llm

# An open instruction-tuned model run locally: no key, no bill. Downloads
# Qwen2.5-7B-Instruct (15.2 GB, Apache-2.0) on first use. Kept out of
# `reproduce` so that reproducing the paper never needs the download.
llm-baseline-local:
	$(PYTHON) -m experiments.run_llm_baseline

study:
	$(PYTHON) -m experiments.run_description_study
	$(PYTHON) -m experiments.run_description_study --encoder paraphrase-multilingual-mpnet-base-v2
	$(PYTHON) -m experiments.run_language_match --no-seeds
	$(PYTHON) -m experiments.run_replication
	$(PYTHON) -m experiments.run_reuters
	$(PYTHON) -m experiments.lexical_gap
	$(PYTHON) -m experiments.run_gap_analysis
	$(PYTHON) -m experiments.run_predictor_search
	$(PYTHON) -m experiments.robustness
	$(PYTHON) -m experiments.run_rocchio
	$(PYTHON) -m experiments.run_rocchio --study definitions

rocchio-preregister:
	$(PYTHON) -m experiments.run_rocchio --preregister

rocchio:
	$(PYTHON) -m experiments.run_rocchio

rocchio-definitions-preregister:
	$(PYTHON) -m experiments.run_rocchio --study definitions --preregister

rocchio-definitions:
	$(PYTHON) -m experiments.run_rocchio --study definitions

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

submission: paper
	$(PYTHON) -m experiments.submission

demo:
	$(PYTHON) tools/make_demo_gif.py

clean:
	rm -rf results/*.json results/*.csv results/tables/*.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
