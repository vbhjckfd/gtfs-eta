# Model-search journal, runs 38–43 (archived in run 45)

## 2026-10-04 — run 38 (early exit, no fresh day set)

Lock taken at 12:14 UTC. Main had not moved. R2 credentials present. Newest possible complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification. No housekeeping due (last done run 35).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.

## 2026-10-04 — run 39 (early exit, no fresh day set)

Lock taken at 16:14 UTC. Main had not moved. R2 credentials present. Newest possible complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification. Housekeeping due next run (run 40, 5th since run 35).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.

## 2026-10-04 — run 40 (housekeeping; early exit, no fresh day set)

Lock taken at 20:13 UTC. Main had not moved. R2 credentials present. Newest possible complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification.
**Housekeeping** (5th run since run 35, own commit): runs 33–37 → `archive/JOURNAL-runs-33-37.md`; arms.py drops sc_big_fo_novl and sc_big_fo_pax
(each lost on ds4 and ds4shift; nothing imports them; _ODOX kept for sc_big_fo_all; report.py keeps their descriptions for the archived rows).
Only three .sh files remain (ds.sh, run.sh, queue.sh), so no runner pruning. Checks: `python report.py` regenerates LEADERBOARD.md unchanged;
baseline and sc_big_fo_duty fit and predict on a synthetic 3000-row frame (no data on this box).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.
2. Next housekeeping due run 45.

## 2026-10-05 — run 41 (early exit; hand-off landed on main)

Lock taken at 00:14 UTC. R2 credentials present. Main moved: it now ships `live_v2` = this branch's `sc_big_fo_long` (see "Production status").
Merged origin/main into the branch (clean). No experiment: the newest complete day is 10-04, and every fresh-split idea now has to beat live_v2 by ≥ 3%,
which no recorded candidate does. No result rows, no notification.

### Next steps
1. **Recommend pausing the schedule** (owner's call): the search's hand-off is in production, and the remaining ideas need new data (operator break
   schedules for 133 / 137 / 122) rather than features.
2. If it continues: from ≥ 10-12, build a fresh day set with the production pipeline (`make pipeline`; `python -m src.train_live --train … --test …`)
   to measure live_v2 on unseen days, then try `_PA` + `_DUTY` ported into src/live_features.py behind a flag (expected ≈ −1%, below the win bar;
   only worth it as a cheap optional add-on).
3. Next housekeeping due run 45.

## 2026-10-05 — run 42 (early exit, no fresh day set)

Lock taken at 04:13 UTC. Main had not moved since run 41. R2 credentials present. Newest complete day is 10-04; a fresh split needs ≥ 10-12
(state step 3 / run 41 step 2), so no experiment, no result rows, no notification.

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC).
2. Next housekeeping due run 45.

## 2026-10-05 — run 43 (early exit, no fresh day set)

Lock taken at 08:14 UTC. Main had not moved since run 41. R2 credentials present. Newest complete day is 10-04; a fresh split needs ≥ 10-12
(state step 3 / run 41 step 2), so no experiment, no result rows, no notification.

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC).
2. Next housekeeping due run 45.

