#!/bin/sh
# run 11 part 2: 127-leaf sc_big_lap on both splits.
export MS_FEAT_DIR=ms_features_v7
Q=research/model-search/queue.sh
save() {
  git add research/model-search/results >/dev/null 2>&1
  git commit -qm "model-search run 11: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
sh $Q ds1v7 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 sc_lap_127; save sc_lap_127
sh $Q ds1shiftv7 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 sc_lap_127; save shift_127
echo RUN11B DONE
