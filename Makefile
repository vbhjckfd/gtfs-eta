.PHONY: install pipeline pipeline-date train learn export deploy \
        sanity validate-off-route validate-horizon check-gtfs check-snapshots \
        push-feed serve-feed smoke score review-quality diagnose help

# Days processed concurrently by `make pipeline` / `make learn`.
# Each worker peaks at ~2 GB — keep ≤ 4-5 on a 16 GB machine.
PARALLEL ?= 4

# ── Setup ──────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

# ── Data pipeline ──────────────────────────────────────────────────────────

## Run the full pipeline for all available days (skips already-done days)
pipeline:
	python scripts/run_pipeline.py --all --parallel $(PARALLEL)

## Run the pipeline for a single date: make pipeline-date DATE=2026-06-01
pipeline-date:
	python scripts/run_pipeline.py --date $(DATE)

# ── Model ──────────────────────────────────────────────────────────────────

## Build features from data/training/ and train the sklearn model
train:
	python -m src.train data/training/

## Full learning cycle: regenerate training data (parallel) + train the model
learn: pipeline train

## Report held-out bias + MAE per stops_ahead (curve should stay flat, near zero)
validate-horizon:
	python scripts/validate_horizon_bias.py

# ── Cloudflare Worker ──────────────────────────────────────────────────────

## Serialise GTFS + model and upload both to R2
export:
	python scripts/export_worker_data.py

## Deploy the Cloudflare Worker (requires Node >=22)
deploy:
	cd worker && bash -c 'source ~/.nvm/nvm.sh && nvm use 24 && \
		npx wrangler deploy --var GIT_COMMIT:$$(git rev-parse --short HEAD)'

## Export then deploy in one step
release: export deploy

# ── Validation & checks ────────────────────────────────────────────────────

## Push one TripUpdates snapshot to R2 (requires exported data in R2)
push-feed:
	python scripts/push_feed.py

## Push to R2 continuously every 10 s (keeps the live feed fresh)
serve-feed:
	python scripts/push_feed.py --loop 10

## Smoke-test the live worker as a drop-in TripUpdates feed.
## Override target with SMOKE_URL=... (e.g. a preview deploy or local dev).
smoke:
	pytest tests/test_smoke.py -v

## Score live ETA quality for a day: make score DATE=2026-06-15 [OUT=report.json]
score:
	python -m src.scoring --date $(or $(DATE),$(shell date -u +%F)) $(if $(OUT),--out $(OUT),)

## Gather a day's quality report + issue notes for review (default: yesterday)
review-quality:
	python scripts/review_quality.py $(if $(DATE),--date $(DATE),)

## On-demand AI diagnosis of a scored day, posted to the issue (needs .[ai])
diagnose:
	python scripts/diagnose.py $(if $(DATE),--date $(DATE),)

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
	@echo "  pipeline             Run pipeline for all days (incremental, PARALLEL=$(PARALLEL))"
	@echo "  pipeline-date DATE=  Run pipeline for a single date"
	@echo "  train                Build features + train model"
	@echo "  learn                pipeline + train in one step"
	@echo "  validate-horizon     Held-out bias + MAE per stops_ahead (post-retrain check)"
	@echo ""
	@echo "  export               Upload GTFS + model to R2"
	@echo "  deploy               Deploy Cloudflare Worker"
	@echo "  release              export + deploy"
	@echo ""
	@echo "  push-feed            Push one TripUpdates snapshot to R2"
	@echo "  serve-feed           Push to R2 every 10 s (live feed daemon)"
	@echo "  smoke                Smoke-test the live worker feed"
	@echo "  score DATE=          Score live ETA quality for a day (default: today)"
	@echo "  review-quality DATE= Gather a day's report + issue notes for review (default: yesterday)"
	@echo "  diagnose DATE=       On-demand AI diagnosis posted to the issue (default: yesterday)"
	@echo "  sanity               Check R2 collection health"
	@echo "  validate-off-route   Validate off-route detection"
	@echo "  check-gtfs           Test GTFS static loading"
	@echo "  check-snapshots      Test R2 snapshot reading"
	@echo ""
