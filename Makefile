.PHONY: install pipeline pipeline-date train export deploy \
        sanity validate-off-route check-gtfs check-snapshots \
        push-feed serve-feed smoke help

# ── Setup ──────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

# ── Data pipeline ──────────────────────────────────────────────────────────

## Run the full pipeline for all available days (skips already-done days)
pipeline:
	python scripts/run_pipeline.py --all

## Run the pipeline for a single date: make pipeline-date DATE=2026-06-01
pipeline-date:
	python scripts/run_pipeline.py --date $(DATE)

# ── Model ──────────────────────────────────────────────────────────────────

## Build features from data/labeled/ and train the sklearn model
train:
	python -m src.train data/labeled/

# ── Cloudflare Worker ──────────────────────────────────────────────────────

## Serialise GTFS + model and upload both to R2
export:
	python scripts/export_worker_data.py

## Deploy the Cloudflare Worker (requires Node >=22)
deploy:
	cd worker && bash -c 'source ~/.nvm/nvm.sh && nvm use 24 && npx wrangler deploy'

## Export then deploy in one step
release: export deploy

# ── Validation & checks ────────────────────────────────────────────────────

## Push one TripUpdates snapshot to R2 (requires exported data in R2)
push-feed:
	python scripts/push_feed.py

## Push to R2 continuously every 30 s (keeps the live feed fresh)
serve-feed:
	python scripts/push_feed.py --loop 30

## Smoke-test the live worker as a drop-in TripUpdates feed.
## Override target with SMOKE_URL=... (e.g. a preview deploy or local dev).
smoke:
	pytest tests/test_smoke.py -v

## Sanity-check R2 collection health
sanity:
	python scripts/sanity_check.py

## Validate off-route detection on a real day's data
validate-off-route:
	python scripts/validate_off_route.py

## Test GTFS static loading
check-gtfs:
	python scripts/step1_load_gtfs.py

## Test snapshot reading from R2
check-snapshots:
	python scripts/step2_verify_snapshots.py

# ── Help ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  install              Install Python dependencies"
	@echo ""
	@echo "  pipeline             Run pipeline for all days (incremental)"
	@echo "  pipeline-date DATE=  Run pipeline for a single date"
	@echo "  train                Build features + train model"
	@echo ""
	@echo "  export               Upload GTFS + model to R2"
	@echo "  deploy               Deploy Cloudflare Worker"
	@echo "  release              export + deploy"
	@echo ""
	@echo "  push-feed            Push one TripUpdates snapshot to R2"
	@echo "  serve-feed           Push to R2 every 30 s (live feed daemon)"
	@echo "  smoke                Smoke-test the live worker feed"
	@echo "  sanity               Check R2 collection health"
	@echo "  validate-off-route   Validate off-route detection"
	@echo "  check-gtfs           Test GTFS static loading"
	@echo "  check-snapshots      Test R2 snapshot reading"
	@echo ""
