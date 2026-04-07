.PHONY: test lint train data-gen firmware-build serve-api serve-web clean

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ apps/ scripts/ tests/
	uv run ruff format --check src/ apps/ scripts/ tests/

train:
	uv run python -m regoscan.train --model cnn --data data/synth_v1.csv --epochs 50 --out runs/cnn_v2/ --cv-folds 5

data-gen:
	uv run python scripts/generate_synth_dataset.py --n-samples 400 --measurements-per-sample 10 --out data/synth_v1.csv

firmware-build:
	cd firmware && pio run

serve-api:
	uv run uvicorn apps.api:app --host 127.0.0.1 --port 8000 --reload

serve-web:
	cd web && npm run dev

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
