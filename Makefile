.PHONY: all data features train test clean model check-model ex4-valid ex4-test compare

# --- The pipeline ----------------------------------------------------------
all: data features train

data:
	python -m src.ingest

features:
	python -m src.features

train:
	python -m src.train

test:
	python -m pytest tests/ -q

clean:
	rm -f data/processed/*.parquet reports/metrics.json reports/*.png

# --- Exercises 4 and 5 (not part of `make all`) ------------------------------
# metrics.json stays the untransformed pipeline; these write elsewhere.
ex4-valid:
	for t in ratio logratio diff; do \
	  python -m src.train --target $$t --splits valid --out reports/exercise4/valid_$$t.json; \
	done

ex4-test:
	python -m src.train --target ratio --out reports/exercise4/test_ratio.json

compare:
	python -m src.compare

# --- The agent model (open weights, local) ----------------------------------
# Builds `bikeshare-coder`: the base model plus a 64k context window, which
# Claude Code needs and Ollama does not give you by default. Pick a larger
# BASE if your machine has the memory (see README.md).
BASE ?= qwen3.5:9b

# Ollama 0.40 does not read a Modelfile from stdin (`-f -` fails with "no
# Modelfile or safetensors files found"), so write it to a temp file.
model:
	ollama pull $(BASE)
	tmp=$$(mktemp) && sed "s|^FROM .*|FROM $(BASE)|" model/Modelfile > $$tmp && \
	  ollama create bikeshare-coder -f $$tmp; status=$$?; rm -f $$tmp; exit $$status

check-model:
	bash model/check.sh
