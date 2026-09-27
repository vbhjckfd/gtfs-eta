#!/bin/sh
# run 16: longer training window (DS2). Pipeline days 08-17..09-21, prep all into v10.
# ds1v10f: ds1 re-fit; its 09-07..09-13 train days now get a full 14-day history (was 7-13 days), then train 08-31..09-18 (19 days, full 14-day
# history) at 1% (window + volume) and at 0.63% (same rows as ds1: window only).
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
D=research/model-search
H=$D/harness.py
RUN=16 DAYS=2026-08-17..2026-09-21 PREP_DIR=ms_features_v10 TAG=v10f \
  ARMS_A="${ARMS_A-baseline sc_big_dwell}" ARMS_B="" sh $D/run.sh || exit 1
save() {
  git add $D/results >/dev/null 2>&1
  git commit -qm "model-search run 16: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
export MS_FEAT_DIR=ms_features_v10 MS_INPLACE=1
for a in baseline sc_big_dwell; do
  python $H run --arm $a --train 2026-08-31..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds2w --keep-pct 0.63; save "ds2w $a"
done
for a in baseline sc_big_dwell; do
  python $H run --arm $a --train 2026-08-31..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds2; save "ds2 $a"
done
echo RUN16 DONE
