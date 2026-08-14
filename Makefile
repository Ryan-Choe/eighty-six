.DEFAULT_GOAL := help
PY := .venv/bin/python

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## venv (python 3.12 via uv) + deps + .env template
	uv venv --python 3.12
	uv pip install -r requirements.txt
	cp -n .env.example .env || true

lock:  ## freeze exactly what's installed
	uv pip freeze > requirements.lock

seed:  ## reset SQLite + rebuild the Chroma index (also the demo reset button)
	$(PY) scripts/seed_db.py
	$(PY) scripts/ingest_kb.py

simulate:  ## run the Friday-rush order batch through ingestion
	$(PY) -m eightysix.ingest data/pos_orders/orders_friday.json

run:  ## Streamlit UI
	.venv/bin/streamlit run app.py

cli:  ## terminal REPL (dev harness / fallback demo)
	$(PY) cli.py

test:  ## deterministic tests — no API keys needed
	$(PY) -m pytest -q

eval:  ## upload datasets + run LangSmith experiments
	$(PY) evals/build_datasets.py
	$(PY) evals/run_evals.py

demo:  ## curated demo runs into the eighty-six-demo LangSmith project
	LANGSMITH_PROJECT=eighty-six-demo $(PY) scripts/demo_runs.py
