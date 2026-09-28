#!/bin/sh
# Parameterised run driver (replaces the per-run runNN.sh scripts).
#   RUN=15 PREP_DIR=ms_features_v10 FEAT_DIR=ms_features_v12 ADDON=addon_v12.py TAG=v12 \
#   ARMS_A="baseline sc_big_dwell ..." ARMS_B="baseline sc_big_dwell ..." sh run.sh
# 1. prep chain into $PREP_DIR as pipeline_lite days land (skip with PREP_DIR=);
# 2. optional add-on script ($ADDON --days ...) writing $FEAT_DIR;
# 3. fits on ds1$TAG (ARMS_A) and ds1shift$TAG (ARMS_B) with $FEAT_DIR;
# 4. optional prep variants (replaces run17/run19.sh): VARIANTS="r32:MS_HIST_RECENT=3,MS_HIST_RECENT_REP=2 ..."
#    re-preps 09-07..09-21 into ${PREP_DIR}_<name> with those env vars (prior days' links_/runs_ symlinked
#    from $PREP_DIR) and fits $ARMS_V (default sc_big_dwell) on tag ds1<name>.
# Results are committed and pushed after every fit so a reclaimed session keeps them.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
D=research/model-search
H=$D/harness.py
DAYS=${DAYS:-2026-08-31..2026-09-21}
save() {
  git add $D/results >/dev/null 2>&1
  git commit -qm "model-search run $RUN: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
if [ -n "$PREP_DIR" ]; then
  for day in $(python -c "import sys;sys.path.insert(0,'$D');from harness import expand_days;print(' '.join(expand_days('$DAYS')))"); do
    while [ ! -f data/ms_lite/$day.pos.parquet ] || [ ! -f data/ms_lite/$day.parquet ]; do sleep 30; done
    sleep 20
    while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 6 ]; do sleep 30; done
    MS_FEAT_DIR=$PREP_DIR python $H prep --days $day || { echo "PREP FAIL $day"; exit 1; }
    echo "PREP OK $day $(date -u +%H:%M)"
  done
  echo PREP DONE
fi
while pgrep -f pipeline_lite.py >/dev/null; do sleep 30; done
if [ -n "$ADDON" ]; then
  python $D/$ADDON --days ${ADDON_DAYS:-2026-09-07..2026-09-21} || { echo "ADDON FAIL"; exit 1; }
  echo ADDON DONE
fi
export MS_FEAT_DIR=${FEAT_DIR:-$PREP_DIR}
mkdir -p data/ms_models
for a in $ARMS_A; do
  python $H run --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 --seed 42 --tag ds1$TAG
  save "ds1$TAG $a"
done
for a in $ARMS_B; do
  python $H run --arm $a --train 2026-09-07..2026-09-17 --test 2026-09-18..2026-09-20 --seed 7 --tag ds1shift$TAG
  save "ds1shift$TAG $a"
done
for V in $VARIANTS; do
  NAME=${V%%:*}; ENVS=$(echo "${V#*:}" | tr ',' ' ')
  FD=${PREP_DIR}_$NAME; mkdir -p data/$FD
  for f in data/$PREP_DIR/links_*.parquet data/$PREP_DIR/runs_*.parquet; do
    case $f in *2026-09-0[7-9]*|*2026-09-[12]*) continue;; esac
    ln -sf "$(pwd)/$f" data/$FD/
  done
  for day in $(python -c "import sys;sys.path.insert(0,'$D');from harness import expand_days;print(' '.join(expand_days('2026-09-07..2026-09-21')))"); do
    [ -f data/$FD/$day.parquet ] && continue
    env $ENVS MS_FEAT_DIR=$FD python $H prep --days $day || { echo "PREP FAIL $NAME $day"; exit 1; }
  done
  for a in ${ARMS_V:-sc_big_dwell}; do
    MS_FEAT_DIR=$FD python $H run --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 --seed 42 --tag ds1$NAME
    save "ds1$NAME $a"
  done
done
echo "RUN $RUN DONE"
