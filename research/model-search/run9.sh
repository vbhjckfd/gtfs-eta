#!/bin/sh
# run 9: prep chain (V7 lap cols, new feature dir) as pipeline_lite days land, then fits.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
export MS_FEAT_DIR=ms_features_v7
H=research/model-search/harness.py
Q=research/model-search/queue.sh
for day in $(python -c "import sys;sys.path.insert(0,'research/model-search');from harness import expand_days;print(' '.join(expand_days('2026-08-31..2026-09-21')))"); do
  while [ ! -f data/ms_lite/$day.pos.parquet ] || [ ! -f data/ms_lite/$day.parquet ]; do sleep 30; done
  sleep 20
  while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 6 ]; do sleep 30; done
  python $H prep --days $day || { echo "PREP FAIL $day"; exit 1; }
done
echo PREP DONE
while pgrep -f pipeline_lite.py >/dev/null; do sleep 30; done
mkdir -p data/ms_models
MS_SAVE_MODEL=data/ms_models/baseline.pkl sh $Q ds1v7 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 baseline
MS_SAVE_MODEL=data/ms_models/sc_big_cur.pkl sh $Q ds1v7 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 sc_big_cur
sh $Q ds1v7 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 sc_big_lap sc_big_cur_reg
sh $Q ds1shiftv7 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 baseline sc_big_cur sc_big_lap sc_big_cur_reg
echo RUN9 DONE
