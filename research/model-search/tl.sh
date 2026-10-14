#!/bin/sh
# Run 44+: arms on the production trainer (src/train_live.py), one fit per call.
#   NAME=<arm>_<split>_s<seed> TRAIN=a..b TEST=c..d SEED=42 [EXTRA=1] [DROP=c1,c2] [CMP=1] sh research/model-search/tl.sh
# EXTRA=1 sets GTFS_ETA_LIVE_EXTRA (duty_* + pa_* cols, src/live_features.py); DROP leaves cols out of the model.
# All arms read one feature cache (built with EXTRA=1, a superset) so rows are identical across arms.
# Appends {"name": NAME, ...train_live result...} to research/model-search/results_tl.jsonl. CMP=1 also fits the legacy recipe.
set -e
D=research/model-search
TMP=$(mktemp)
export GTFS_ETA_LIVE_FEAT_DIR=${FEAT:-data/features_live_x}
export GTFS_ETA_LIVE_EXTRA=${EXTRA:-0} GTFS_ETA_LIVE_DROP=${DROP:-}
python -m src.train_live --train "$TRAIN" --test "$TEST" --seed "${SEED:-42}" --parallel "${PAR:-3}" \
  --no-save --results "$TMP" ${CMP:+--compare-baseline}
python - "$TMP" "$NAME" "$D/results_tl.jsonl" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
r["name"] = sys.argv[2]
open(sys.argv[3], "a").write(json.dumps(r) + "\n")
PY
rm -f "$TMP"
