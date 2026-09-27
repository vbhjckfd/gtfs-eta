#!/bin/sh
# run 14 add-on: V11 dwell cols into ms_features_v11 (from v10 files), then fits
# after run14.sh finishes.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
H=research/model-search/harness.py
Q=research/model-search/queue.sh
save() {
  git add research/model-search/results >/dev/null 2>&1
  git commit -qm "model-search run 14: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
python research/model-search/addon_v11.py --days 2026-09-07..2026-09-21 || { echo "ADDON FAIL"; exit 1; }
echo ADDON DONE
while pgrep -f run14.sh >/dev/null; do sleep 30; done
export MS_FEAT_DIR=ms_features_v11
A="2026-09-07..2026-09-18 2026-09-19..2026-09-21 42"
B="2026-09-07..2026-09-17 2026-09-18..2026-09-20 7"
sh $Q ds1v11 $A baseline sc_big_dwell; save v11_base
sh $Q ds1v11 $A sc_big_dwell_h sc_big_dwell_v11; save v11_ds1
sh $Q ds1shiftv11 $B baseline sc_big_dwell_h sc_big_dwell_v11; save v11_shift
echo RUN14B DONE
