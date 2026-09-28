#!/bin/sh
# run 19: recency weighting inside the leader's static history tables (hist link x hour, weekday/weekend,
# location dwell). Prep 08-31..09-21 into v10 (tag v19: baseline + leader), then for each R:REP re-prep
# 09-07..09-21 with MS_HIST_RECENT=R MS_HIST_RECENT_REP=REP (the last R prior days count REP times,
# 14-day lookback unchanged; prior-day links_/runs_ symlinked from v10) and fit the leader on ds1r$R$REP.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
D=research/model-search
H=$D/harness.py
RUN=19 PREP_DIR=ms_features_v10 TAG=v19 \
  ARMS_A="${ARMS_A-baseline sc_big_dwell}" ARMS_B="" sh $D/run.sh || exit 1
save() {
  git add $D/results >/dev/null 2>&1
  git commit -qm "model-search run 19: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
for RR in ${RRS-3:2 7:2}; do
  R=${RR%:*}; REP=${RR#*:}
  FD=ms_features_v10r$R$REP; mkdir -p data/$FD
  for f in data/ms_features_v10/links_2026-0[89]-*.parquet data/ms_features_v10/runs_2026-0[89]-*.parquet; do
    case $f in *2026-09-0[7-9]*|*2026-09-[12]*) continue;; esac
    ln -sf "$(pwd)/$f" data/$FD/
  done
  for day in $(python -c "import sys;sys.path.insert(0,'$D');from harness import expand_days;print(' '.join(expand_days('2026-09-07..2026-09-21')))"); do
    [ -f data/$FD/$day.parquet ] && continue
    MS_HIST_RECENT=$R MS_HIST_RECENT_REP=$REP MS_FEAT_DIR=$FD python $H prep --days $day || { echo "PREP FAIL r$RR $day"; exit 1; }
    echo "PREP OK r$RR $day $(date -u +%H:%M)"
  done
  MS_FEAT_DIR=$FD python $H run --arm sc_big_dwell --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds1r$R$REP
  save "ds1r$R$REP sc_big_dwell"
done
echo RUN19 DONE
