# Makefile for quick commands

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest --cov=api --cov=hierarchical_lm --cov-report=term-missing

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py

docker-build:
	docker build -t hlm:latest .

docker-run:
	docker run -p 8000:8000 hlm:latest

lint:
	ruff check api hierarchical_lm benchmarks tests

benchmark:
	python benchmarks/benchmark_hlm.py --iterations 100 --output benchmark-results.json
	python -m json.tool benchmark-results.json > /tmp/benchmark-results.pretty.json

.PHONY: ingest api dash

# Build vector index from docs/ and data/
ingest:
	python -m src.ingest --folders data docs

# Run FastAPI API
api:
	uvicorn api.main:app --reload --port 8080

# Run Streamlit dashboard
dash:
	streamlit run dashboard/rag_app.py
