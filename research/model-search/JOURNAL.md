# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results JSON per arm in `results/`; `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries 1–12 are in `archive/JOURNAL-runs-1-12.md`.

## State of the search (updated run 17)

### Protocol set-up (fixed)
- Fresh box: `pip install -e . tzdata` (without tzdata every pipeline day fails).
- Rebuild: `python research/model-search/pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21`
  (≈ 1.5–2.25 h; 10% of snapshots + full-day `.cross` / `.pos` side tables; peak ~11 GB).
  `run_pipeline.py` itself OOMs at `--parallel 2+` on this 4-core / 15.7 GB box.
- Prep: `harness.py prep --days D` per day, sequential (history lookups), ≈ 2.5 min/day,
  into `data/$MS_FEAT_DIR`. The prep chain + fits are driven by `run.sh` (parameterised).
- Splits: `ds1` = train 09-07..09-18, test 09-19 Sat / 09-20 Sun / 09-21 Mon, seed 42;
  `ds1shift` = train 09-07..09-17, test 09-18 Fri / 09-19 / 09-20, seed 7. Train 1% of
  snapshots/day (≈2.18M / 1.98M rows), test 3% (≈1.41M / 1.42M rows).
  08-31..09-06 only feed the 14-day history tables. Baseline = `src/train.py` recipe:
  110.6 / p90 230.9 (ds1), 113.0 / p90 241.1 (shift); reproduces bit-for-bit every run.
- Fits: ~5 min baseline, ~10–15 min for 255-leaf leaders; never fit while the pipeline runs.
  Background loop refreshes LOCK every 50 min (runs last > 3 h).

### Current leader: `sc_big_dwell` (run 14) — WIN on both splits/seeds
ds1 110.6 → 87.0 (−21.3%), p90 230.9 → 178.2; shift 113.0 → 87.0 (−23.1%), p90 241.1 → 179.5;
every stops_ahead bucket better (ds1 sa1 72.6→60.7, sa10 153.5→121.6). Confirmed at 3x train data
(run 15, `ds1v12k3`): 108.1 → 84.7 (−21.7%), p90 226.9 → 173.3.
Recipe (arms.py `sc_big_dwell`): production FEATURE_COLS − {month, day_of_week, stop_sequence},
route_id and target-stop code (top-254 stops) as native categoricals, 255 leaves /
min_samples_leaf 50, lr 0.05, 1200 iters (cap binds), sentinel −1 for missing live values, plus:
1. live link store: per stop-pair link the last 5 traversals (≤ 30 min, 30 s detect lag) →
   path sums with median-of-3 / median-of-5, coverage, age, speed (`_STACK`);
2. own speed over 60/180/300 s (5-min position ring);
3. historical link × hour median tables (all days + weekday/weekend) from the prior 14 days
   (`_DT`), built at export like the priors;
4. current-link live m5 / hist / fraction left (`_CUR`);
5. own previous-lap link times, 3 h window (`_LAP`);
6. vehicle-level own/hist link-time ratio over the last hour, across trips (`_VEH`);
7. conditional remaining dwell at the current location (last passed stop, 50 m bin) given
   stationary_sec, from the prior 14 days' stationary runs (`_DWELL`).
**Serving status: needs serving work, ≈ 3 days** (live link store + position ring + per-vehicle
crossing log persisted in `feed/tracker_state.json` across the 5-min push-feed restarts;
export of categorical bitsets / missing_go_to_left; static hist + dwell tables at export).
Timing (run 11): 255 leaves + bitsets = 1.55 s per 3000-row batch single-threaded (baseline
0.75 s), export ~29 MB — fits the 10 s cycle. Productionising is the owner's call.

### What won (in order; each Δ is vs the previous leader, both splits)
- live link path times (latest traversal per link): −7 to −9% vs baseline (run 1).
- + own speed, − headway / since_last_at_target: −8.3% / −9.0% (run 2).
- median of last 3/5 traversals, drop calendar cols (`stack`): −12.4% / −14.5% (run 3).
- historical link × hour table (`stack_hist`): ≈ −2% (run 4).
- weekday/weekend table + route_id categorical (`stack_hist_dt_cat`): ≈ −2.3% (run 6).
- target-stop categorical + 255 leaves (`dtcat_stopcat_big`): −1.6% (run 7).
- current-link cols (`sc_big_cur`): −0.6 to −1.1% (run 8).
- own previous lap (`sc_big_lap`): −1.2 to −1.5% (run 11).
- vehicle-level own/hist ratio (`sc_big_veh`): −1.0 to −1.2% (run 12).
- location dwell table (`sc_big_dwell`): −0.8% (run 14).

### Ruled out (one line each)
- Segment-additive link-time model: loses to direct HGBT at every horizon (repo history).
- Lookup-table bias correction: hurts MAE (repo history).
- Headway via since_last_at_target (any route): noise (run 1–2).
- NaN for missing live values: worse than sentinel −1 (exported trees route NaN right) (run 1).
- Cold-start dropout: costs warm accuracy; persist the live state instead; gate not needed (runs 2–3).
- lr 0.1 / lr 0.08, min_samples_leaf 100, l2 1, leaf100: neutral (runs 2, 6, 8, 11).
- No hour sample weights: neutral (run 3). Loss: prod already uses absolute_error (run 11).
- m7, EWMA per link, p75 table, live/hist ratio, fill path: ≤ 0.3% (runs 4, 6).
- Own-vs-hist ratio keyed by trip, recency weighting, pruning 3 cols: noise/loss (run 7).
- Residual target (base ETA + tree) and log-ratio target: no gain (run 7).
- Next-stop categorical: noise, p90 worse (run 8).
- 127 leaves for the lap model: loses most of the lap gain (run 11).
- 6 h lap window, 20 min / clipped vehicle ratio, network-wide 15-min ratio: ±0.2% (runs 12–13).
- Dwell table keyed by 3-hour band, live same-location last-3 dwell: +0.1–0.2% (run 14).
- Route 122 (230–330 s): data-sparse (2 vehicles, 43% live coverage, long links); untouched by
  every feature family; more training data helps it most (3x: 325→296). Treat as data issue.

- Same-route headway to the leader (at the last stop / target, leader's own gap): +0.3% (run 15).
- Stop-keyed dwell fallback for sparse 50 m bins: +0.1% (run 15).

- Longer train window (19 vs 12 days) is a data lever, not a model one: at equal rows both the baseline and
  the leader gain ≈ 0.8–1.5%, so the leader's relative gain is unchanged (−20.7% vs −21.3%) (run 16).
- History-table lookback (hist link × hour + dwell tables): 14 d is best; 21 d +0.2% (noise), 7 d +0.9% (run 17).

### Open next steps
1. Longer window / more rows help both models equally (run 16), and production already trains on all days.
   The model-side search has saturated. The remaining open item is hand-off.
2. Route 133 trip-structure diagnostic (headway cols moved only this route).
3. Hand-off of sc_big_dwell is the owner's call. Keep the 14-day lookback for its static tables (run 17).

## 2026-09-27 — run 13 (network-wide ratio, short / clipped vehicle ratios)

Lock taken at 03:17 UTC. Rebuilt DS1 with `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21`,
at about 41 min per batch of 4 days, then ran `sh research/model-search/run13.sh`, which runs the prep chain
into `ms_features_v9` and then the fits. A background loop refreshes LOCK every 50 min, because the
run lasts longer than the 3 h lock age.

New V9 columns in `features_live.py` (V8 outputs unchanged):
- `net_ratio15`, `net_n15`: network-wide sum(link time, clipped to [hist/3, 3·hist]) / sum(historical
  day-hour median) over *all* vehicles' links completed in the last 15 min (citywide traffic state; a
  global cumsum plus searchsorted, O(1) per row when serving). On 09-01 it ranges 0.77–1.24 (median 1.04),
  with n ≈ 3k links per 15 min.
- `veh_hist_ratio20`: the run-12 vehicle-level ratio over a 20 min window.
- `veh_hist_ratio60c`: the 60 min vehicle ratio with each link clipped to [hist/3, 3·hist], so that
  layover and deadhead links at the terminus can't dominate it (run 12 next step 1, the route 133 regression).

Arms: `sc_big_net` (sc_big_veh + net), `sc_big_v9` (net + ratio20 + ratio60c).

The rebuild ran from 03:20 to 05:36 UTC (first batch about 41 min per day, later days about 20 min).
Baseline and sc_big_veh reproduced exactly (110.6 / 113.0 and 87.7 / 87.7).

### Results (MAE / p90; Δ vs baseline, Δ vs the leader sc_big_veh)
| arm | ds1v9 (s42) | Δbase | Δleader | ds1shiftv9 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| sc_big_veh (leader) | 87.7 / 179.1 | −20.7% | | 87.7 / 180.6 | −22.4% | |
| sc_big_net | 87.8 / 179.0 | −20.6% | +0.07% | 87.5 / 179.8 | −22.5% | −0.13% |
| sc_big_v9 | 87.6 / 178.1 | −20.8% | −0.13% | 87.5 / 179.4 | −22.6% | −0.16% |

- Both are noise-level. sc_big_v9 trims p90 by 0.5–0.65% and the long horizons (ds1 sa10
  122.3→121.6), but sa1 gets 0.5–0.9% worse. The network ratio alone does nothing: the
  shared live link store already carries current traffic on each link, so a citywide scalar adds
  no information.
- The clipped vehicle ratio does not fix route 133 on shift (146.0 → 146.9), so the run-12
  regression there is not caused by layover links. Route 122 gets worse with the net
  cols (ds1 325→334–337, shift 230→246–247). 122 is sparse and long-headway, so a global
  scalar is a spurious split for it.
- Rows: train 2.18M / 1.98M, test 1.41M / 1.42M (unchanged).

**Conclusion: no new leader. sc_big_veh stays. The vehicle/network "state ratio" family is
exhausted: 3 h / 6 h lap, 20 / 60 min vehicle ratio, clipped, and network-wide all land within ±0.2%.**

### Scale check: 3x training rows (tag `ds1v9k3`, `--keep-pct 3`, 6.51M train rows, seed 42)
The first attempt was OOM-killed for both arms, because the v9 feature files are too wide for 3x in 15.7 GB.
`harness.py` now reads `MS_DROP_COLS` (a comma list of feature cols to skip at load; numerics
unchanged). run13b.sh drops the 21 live cols that neither arm uses.
| arm | MAE / median / p90 | Δbase | sa1 / sa5 / sa10 | Sat / Sun / Mon | fit |
|---|---|---|---|---|---|
| baseline | 108.1 / 54.4 / 226.9 | | 70.4 / 109.7 / 149.9 | 96.8 / 106.3 / 117.4 | 20 min |
| sc_big_veh | 85.4 / 42.4 / 174.3 | −21.0% | 60.2 / 85.5 / 119.2 | 79.7 / 83.8 / 90.7 | 28 min |
- The baseline reproduces run 7's 3x baseline exactly (108.1 / 226.9). 3x data helps the leader by 2.3 s
  (87.7→85.4), a little more than the baseline's 2.5 s relative to its size. The relative gain holds and grows
  (−20.7% → −21.0%), with p90 −23.2% and the worst bucket −14.5%. Bias is −14.7 (baseline −27.5).
- Worst routes at 3x: 122 296, 133 207, 125 150, 106 143, 137 132, 129 131, 111 126, 113 126,
  131 125, 881 124. Route 122 gains the most from more data (325→296), so part of its error is data
  sparsity in training, not only in the live features.

### Next steps
1. Stop adding live-state ratio features (exhausted, see above). The remaining levers are data volume
   (3x = −2.3 s; production trains on 100% of snapshots over more days, so the served gain is likely
   larger than measured here) and the target side.
2. Target-side / dwell: model arrival at the target as arrival at the previous stop plus the target's
   own link, i.e. the historical dwell at the target stop × hour (a static table like hist_dt, keyed by
   stop) and the observed dwell of the last vehicle at the target stop (live, from `.pos`
   stationary runs near the stop).
3. Route 133 (early by ~200 s at long horizons): check whether its trips have a long layover *inside*
   the trip (a timed point), which the model can't see. If so, add a "timed-point stops between
   vehicle and target" count from historical dwell > 60 s.
4. Serving: sc_big_veh is ready to hand off (≈2.7 days, see the run-12 serving estimate; timing test
   in run 11). A productionisation PR is the owner's call, not this routine's.

## 2026-09-27 — run 14 (dwell at the vehicle's current location)

Lock taken at 08:16 UTC. The fresh box had no deps (`pip install -e . tzdata`, as noted in run 11).
Rebuilt DS1 with `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21` (08:20–10:15, ~35 min
per 4-day batch), and ran `sh research/model-search/run14.sh` (prep chain into `ms_features_v10`, then fits).
A background loop refreshes LOCK every 50 min.

Idea (run 13 next step 2, turned around): the tail is stopped vehicles, and `stationary_sec` tells the
model how long a vehicle has been stopped, but not how long stops *at this place* usually last (a red light
vs a terminus layover vs a timed point). New V10 cols in `features_live.py`:
- `stationary_runs(pos)`: every stationary run of the full day, same 25 m anchor rule as
  `labeling._stationary_seconds` (plus a break on >120 s gaps), keyed by location = (last passed stop id,
  50 m bin of the offset past it). ~480k runs/day, median 18 s, 1.4% longer than 120 s. Saved per day as
  `runs_<day>.parquet` next to `links_<day>.parquet`.
- `dwell_features`: over the prior 14 days' runs at the row's location, conditional on the run lasting
  longer than the row's current `stationary_sec` s: `dwell_rem_med` (median of D − s), `dwell_rem_p75`,
  `dwell_p_more120` (P(D > s + 120 | D > s)), `dwell_n`, `dwell_long_share` (share of runs at the key > 120 s).
  NaN when fewer than 5 runs survive. Coverage on 09-10: 89% of all rows; for rows stopped > 60 s, 72%
  (47% with only one prior day); corr with the target on those rows 0.44.
  Serving: a static table key -> sorted durations built at export (like hist_dt), one searchsorted per row.

Arms: `sc_big_dwell` (sc_big_veh + all 5 cols), `sc_big_dwell2` (+ dwell_rem_med, dwell_p_more120 only).

### Results (MAE / p90; Δ vs baseline, Δ vs the leader sc_big_veh)
| arm | ds1v10 (s42) | Δbase | Δleader | ds1shiftv10 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| sc_big_veh (leader) | 87.7 / 179.1 | −20.7% | | 87.7 / 180.6 | −22.4% | |
| **sc_big_dwell** | **87.0 / 178.2** | **−21.3%** | **−0.86%** | **87.0 / 179.5** | **−23.1%** | **−0.79%** |
| sc_big_dwell2 | 87.2 / 177.7 | −21.1% | −0.61% | 87.0 / 179.4 | −23.0% | −0.75% |

- Baselines and the leader reproduced exactly (110.6 / 113.0, 87.7 / 87.7). Rows unchanged: train
  2.18M / 1.98M, test 1.41M / 1.42M (Sat 422k, Sun 404k, Mon 589k, Fri 599k).
- sc_big_dwell vs the leader improves every stops_ahead bucket on both splits, most at short horizons
  where a remaining dwell is a big part of the ETA: ds1 sa1 −2.1%, sa2 −1.4%, sa5 −0.7%, sa10 −0.5%; shift
  sa1 −1.7%, sa10 −0.8%. Every test day improves (ds1 Sat 82.5→81.5, Sun 85.7→84.9, Mon 92.9→92.3; shift
  Fri 92.5→91.9, Sat 82.5→81.8, Sun 85.9→85.1). Median 43.3→43.1 / 44.1→43.9, bias −14.0→−12.6 / −16.0→−14.6.
- Route 133 on shift (the run-12 regression) 146→138; ds1 210→208. Route 122 unchanged-to-worse
  (ds1 325→332, shift 230→234).
- Worst routes (sc_big_dwell): ds1 122 332, 133 208, 106 161, 125 155, 129 138, 881 129, 111 129, 137 127,
  113 126, 131 124; shift 122 234, 106 167, 125 144, 98 140, 133 138, 137 134, 111 133, 2302 130, 129 129, 88 121.
- It is the first gain above ±0.2% since run 12; small in MAE but consistent on both splits/seeds and in
  every bucket and day. Both fits hit the 1200-iteration cap, like the leader.

**Conclusion: new leader `sc_big_dwell` (ds1 110.6→87.0, −21.3%, p90 230.9→178.2; shift 113.0→87.0,
−23.1%, p90 241.1→179.5), −0.8% vs sc_big_veh on both splits with every bucket better.**

### Add-on: V11 dwell cols (`addon_v11.py`, `run14b.sh`, tag `ds1v11`, same rows)
`addon_v11.py` appends cols to the v10 day files (into `ms_features_v11`) without a full re-prep (~50 s/day):
- `dwell_h_rem_med / dwell_h_p_more120 / dwell_h_n`: the V10 conditional remaining dwell keyed by
  (location, 3-hour local band). For stopped > 60 s rows its corr with the target is 0.47 (V10: 0.44), but
  coverage is 58% (V10: 72%).
- `dwell_live_last / dwell_live_age / dwell_live_m3`: today's most recent completed stop (run ≥ 30 s) by any
  vehicle at the same location, ended ≤ t − 30 s within the last hour, plus the mean of the last 3.
  Coverage for stopped > 60 s rows is 81%, corr 0.23 / 0.28.

| arm (ds1v11, s42) | MAE / p90 | Δ vs sc_big_dwell | sa1 / sa5 / sa10 Δ |
|---|---|---|---|
| baseline | 110.6 / 230.9 | | |
| sc_big_dwell | 87.0 / 178.2 | (reproduced exactly) | |
| sc_big_dwell_h | 87.1 / 177.9 | +0.19% | −0.4 / +0.2 / +0.4% |
| sc_big_dwell_v11 | 87.1 / 177.9 | +0.13% | −0.4 / −0.0 / +0.5% |

Both are noise with a slight loss, so I stopped the shift-split fits for them early (only one split
was run, so they are not claims either way). The prior-day location table already carries the signal; hour
banding thins it, and the last live stop at a location is mostly a different vehicle in a different phase
(33% of live rows lack it and it correlates weakly).

### Serving estimate (sc_big_dwell)
On top of sc_big_veh (~2.7 days): at export, build location → sorted stop-duration arrays from the last
14 days of stationary runs (from the same positions the pipeline already projects; ~10k keys/day, a few MB).
At serving, derive the location key from the tracker's dist_along + the trip profile, then do one
searchsorted per row against the stationary_sec the daemon already computes (`inference.stationary_seconds`).
No new live state. About +0.3 day, so ~3 days total.

### Next steps
1. Hand-off candidate is now `sc_big_dwell`. Productionizing it is the owner's decision.
2. Dwell family: try the location key without the 50 m bin (stop only) for more coverage, or a
   fallback chain bin→stop. Also try conditioning on route (terminus layover lengths are route-specific).
3. Route 122 stays the worst (230–330 s) through every feature family. Treat it as a data / target issue.
4. If time allows, do a 3x-data check of sc_big_dwell (`--keep-pct 3` with `MS_DROP_COLS`, as in run 13).

## 2026-09-27 — run 15 (housekeeping; same-route headway, coarse dwell; 3x check of the leader)

Lock taken at 13:17 UTC. First did housekeeping (JOURNAL > 400 lines): runs 1–12 moved to
`archive/JOURNAL-runs-1-12.md`, a "State of the search" section added, `report.py` now writes
LEADERBOARD.md for the current tags and `archive/LEADERBOARD-old.md` for the rest, and the old
`runNN.sh` scripts were replaced by one parameterised `run.sh`. Arms/feature code was not pruned:
nothing in arms.py has lost twice. Most arms ran on one split only, or lost only against a leader.

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21` (13:18–15:45), then
```
RUN=15 PREP_DIR=ms_features_v10 FEAT_DIR=ms_features_v12 ADDON=addon_v12.py TAG=v12 \
  ARMS_A="baseline sc_big_dwell sc_big_hw sc_big_dwc sc_big_v12" ARMS_B="..." sh research/model-search/run.sh
```
(the prep chain into v10, then `addon_v12.py` into v12, about 1 min/day). A smoke test of the add-on while the pipeline
was running got the 09-01 prep OOM-killed. The chain was restarted and cached days were skipped. Don't run
anything heavy next to pipeline_lite.

New V12 cols (`addon_v12.py`, appended to the v10 day files):
- `hw_ahead_cur`: seconds since the previous *other* vehicle of the same route+direction arrived at the stop this
  vehicle passed last (gap to the leader). `hw_ahead_tgt` is the same measured at the target stop. `hw_ahead_prev`
  is the leader's own gap to its leader (regularity). Only crossings at or before t − 30 s within 2 h are used.
  Coverage is 91–93% / 96–98% / 88–92%. The median gap is ~800 s on weekdays and ~970 s at weekends.
  Serving would be a (route, dir, stop) → last 2 arrivals dict.
- `dwell_c_*`: the V10 conditional remaining dwell keyed by the last passed stop only, as a fallback.
  Coverage of stopped > 60 s rows goes from 0.64–0.78 (V10) to 0.77–0.88.

### Results (ds1v12, seed 42; Δ vs the leader sc_big_dwell)
| arm | MAE / median / p90 | Δleader | sa1 / sa5 / sa10 Δ | Sat / Sun / Mon |
|---|---|---|---|---|
| baseline | 110.6 / 56.0 / 230.9 | | | 100.2 / 107.1 / 120.5 |
| sc_big_dwell (leader, reproduced exactly) | 87.0 / 43.1 / 178.2 | | | 81.5 / 84.9 / 92.3 |
| sc_big_hw (+ headway) | 87.2 / 43.1 / 178.1 | +0.30% | +0.3 / +0.1 / +0.3% | 81.7 / 85.3 / 92.6 |
| sc_big_dwc (+ coarse dwell) | 87.1 / 43.1 / 178.2 | +0.11% | −0.1 / −0.0 / +0.3% | 81.5 / 85.1 / 92.4 |
| sc_big_v12 (both) | 87.3 / 43.1 / 178.2 | +0.33% | +0.2 / +0.3 / +0.5% | 81.7 / 85.1 / 92.7 |

Rows: train 2.18M, test 1.41M. All three arms are a small loss, so I skipped the shift-split fits (no claim either way).
Headway was the one family the search had not tried at route level (run 1 tried "any vehicle at the target").
It does move route 133 (208 → 204 / 201 with v12), but it costs elsewhere. The live link store already sees the
leader's traversals, so the gap itself carries no extra information. The coarse dwell key adds coverage but no
accuracy: the rows that lack a 50 m-bin table are the rarely-stopped ones.

### Scale check: 3x training rows (tag `ds1v12k3`, `--keep-pct 3`, 6.51M train rows, seed 42)
At 3x, sc_big_dwell was first OOM-killed. `_sentinel` copied both whole frames, so `MS_INPLACE=1` now
fills in place instead. The numbers are identical and the peak memory is ~8 GB. `run15b.sh` has the command.
| arm | MAE / median / p90 | Δbase | sa1 / sa5 / sa10 | Sat / Sun / Mon | fit |
|---|---|---|---|---|---|
| baseline | 108.1 / 54.4 / 226.9 | | 70.4 / 109.7 / 149.9 | 96.8 / 106.3 / 117.4 | 21 min |
| **sc_big_dwell** | **84.7 / 42.1 / 173.3** | **−21.7%** | 58.6 / 84.7 / 118.7 | 78.7 / 83.0 / 90.1 | 31 min |
- The baseline reproduces run 13's 3x number exactly. The leader at 3x beats the 3x sc_big_veh (85.4) by 0.8%,
  the same margin as at 1%. So the dwell gain holds with more data, and the relative gain grows slightly
  (−21.3% → −21.7%, p90 −23.6%, worst bucket −16.8%). Bias is −13.4 (baseline −27.5).
- Worst routes at 3x: 122 305, 133 211, 125 153, 106 149, 129 144, 881 124, 111 124, 137 123, 113 121, 131 118.

**Conclusion: no new leader. sc_big_dwell stays, now also confirmed at 3x data (108.1 → 84.7, −21.7%).
Same-route headway and the coarse dwell fallback are ruled out.**

### Next steps
1. Feature families on the path, the vehicle state, the location dwell and the headway are all saturated (±0.3%).
   Remaining levers: training volume/days (3x = −2.3 s; prod trains on more days at 100%), and a longer
   training window. A DS2 with 4 weeks of train days (from 2026-08-17, still ≥ 2026-07-31) would test the
   second, but the rebuild takes ~2x as long (~4 h at `--parallel 4`), so it needs a run with nothing else in it.
2. Route 133 alone: the headway cols help it (−3%). A route-133-only diagnostic of its long-horizon early bias
   would need its trip structure (timed points), not more generic features.
3. The leader is ready for hand-off (~3 days of serving work, see the State section). Productionising is the owner's call.

## 2026-09-27 — run 16 (longer training window, DS2)

Lock taken at 18:17 UTC. No housekeeping was due (the journal was 304 lines, and run 16 is not a 5th run). Main was
already merged. This run tested run-15 next step 1: is the 3x-rows gain about volume or about window length?

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-08-17..2026-09-21` (36 days, 18:18–~21:45). Then
`sh research/model-search/run16.sh`, which runs the prep chain into `ms_features_v10` and then the fits. The
09-01 prep was OOM-killed while running next to the pipeline. It was restarted in a retry wrapper, and cached days
were skipped. The new days 08-17..08-30 mean that every day from 08-31 on has the full 14-day history table.
`harness.py` now also takes `MS_HIST_DAYS` (default 14, so nothing changes by default).

Splits, all tested on 09-19 Sat / 09-20 Sun / 09-21 Mon (1.41M rows at 3%), seed 42:
- `ds1v10f`: ds1 (train 09-07..09-18, 1%). Its early train days now have a full 14-day history instead of 7–13 days.
- `ds2w`: train 08-31..09-18 (19 days) at `--keep-pct 0.63`, so the row count matches ds1 (the window effect only).
- `ds2`: the same 19 days at 1% (window plus volume).

| split | train rows | baseline MAE / med / p90 / bias | sc_big_dwell MAE / med / p90 / bias | Δ | Sat / Sun / Mon (leader) |
|---|---|---|---|---|---|
| ds1v10f | 2.18M | 110.6 / 56.0 / 230.9 / −28.2 | 87.0 / 43.2 / 178.2 / −12.1 | −21.3% | 81.4 / 85.0 / 92.4 |
| ds2w | 2.13M | 108.9 / 54.3 / 227.7 / −31.0 | 86.3 / 43.0 / 177.3 / −11.8 | −20.7% | 80.2 / 83.4 / 92.7 |
| ds2 | 3.37M | 107.4 / 53.6 / 224.3 / −31.1 | 85.3 / 42.5 / 175.1 / −13.2 | −20.5% | 79.2 / 82.7 / 91.5 |
| (run 15, ds1 3x) | 6.51M | 108.1 / 54.4 / 226.9 | 84.7 / 42.1 / 173.3 | −21.7% | 78.7 / 83.0 / 90.1 |

The leader's stops_ahead values are sa1 / sa5 / sa10: 60.6 / 87.0 / 122.0 (ds1v10f), 59.9 / 86.2 / 120.8 (ds2w),
59.3 / 85.3 / 119.4 (ds2). Every protocol check passes on every split (the worst bucket is −16%).
- A full history for the early train days changes nothing (87.01 vs 87.0; the baseline is identical, since it has no
  history features). The run 1–15 ds1 numbers are therefore not biased by the truncated history.
- A longer window at the same row count helps both models: the baseline by −1.5%, the leader by −0.8%. The gain is
  almost entirely on weekends (the leader goes Sat 81.4→80.2, Sun 85.0→83.4, Mon 92.4→92.7), because 19 days hold
  6 weekend days where 12 days hold 4. Adding rows on top (ds2) gains another ~1.1%. Per training row, 3x rows over
  12 days (84.7) beats 1.6x rows over 19 days (85.3) only slightly.
- The leader's relative gain is stable across window and volume (−20.5% to −21.7%), so it is not an artefact of a
  short window.
- Route 133 gets *worse* for the leader with the longer window (206 → 222 at equal rows, 209 at ds2), while the
  baseline improves there (278 → 248). This fits a route whose behaviour changed during late August.
  Worst routes for the leader on ds2: 122 333, 133 209, 125 150, 106 147, 129 136, 881 128, 111 128, 137 125,
  113 121, 131 120.

**Conclusion: no new model candidate. sc_big_dwell stays the leader, and its gain holds with a 19-day window.
A longer window and more rows are data levers that help both models about equally. Production already trains on
all available days, so there is nothing to hand off from this run beyond the confirmation.**

### Next steps
1. The model-side search has saturated: the last five runs landed within ±0.3%, except dwell at −0.8%. Unless the
   owner wants specific directions, the most useful remaining work is hand-off support for sc_big_dwell (a serving
   prototype behind a flag in research/, with the timing checked against the live daemon cycle). That is a
   code-change decision for the owner.
2. Route 133: check the operator's schedule or trip changes around late August (the longer window hurts it).
3. If the search continues, try `MS_HIST_DAYS=28` for the historical link and dwell tables. The DS2 days make that
   possible for test days from 09-14 on (prep into a separate feature dir with the links_/runs_ files symlinked).

## 2026-09-28 — run 17 (history-table lookback: 21 / 7 days)

Lock taken at 23:16 UTC (09-27). No housekeeping due (journal 359 lines; run 15 did it, next at run 20).
Main was already merged. Fresh box: `pip install -e . tzdata`.
Tested run-16 next step 3: the lookback of the static history tables that the leader uses (hist link × hour,
weekday/weekend table, location dwell), which is set by `MS_HIST_DAYS` (default 14).

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-08-17..2026-09-21` (23:17–~03:00), then
`sh research/model-search/run17.sh`:
prep 08-17..09-21 into `ms_features_v10` (14 d) → refit baseline + leader on `ds1v10f`; then, for HD in 21 and 7,
re-prep 09-07..09-21 into `ms_features_v10h$HD` with `MS_HIST_DAYS=$HD` (the prior days' `links_` / `runs_`
files are symlinked from v10, since they don't depend on the lookback) and fit the leader on tag `ds1h$HD`.
Two preps (08-30, 09-12) were OOM-killed next to the pipeline again. The driver was rerun in a retry loop, and cached days were skipped.
The ds1h21 / ds1h7 baseline rows are copies of the ds1v10f baseline: the rows are the same and it has no history cols.

### Results (ds1, seed 42, train 2.18M, test 1.41M rows; Sat 422k / Sun 404k / Mon 589k)
| arm | lookback | MAE / median / p90 / bias | Δbase | Δ vs 14 d | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|---|
| baseline | — | 110.6 / 56.0 / 230.9 / −28.2 | | | 72.6 / 112.0 / 153.5 | 100.2 / 107.1 / 120.5 |
| sc_big_dwell | 14 d (ds1v10f, reproduced exactly) | 87.0 / 43.2 / 178.2 / −12.1 | −21.3% | | 60.6 / 87.0 / 122.0 | 81.4 / 85.0 / 92.4 |
| sc_big_dwell | 21 d (ds1h21) | 87.2 / 43.1 / 178.3 / −11.8 | −21.2% | +0.2% | 60.7 / 87.3 / 122.2 | 81.4 / 85.1 / 92.7 |
| sc_big_dwell | 7 d (ds1h7) | 87.8 / 43.5 / 178.8 / −14.2 | −20.6% | +0.9% | 61.6 / 87.8 / 123.0 | 82.8 / 85.8 / 92.9 |

- 21 d is noise-level worse across the board. 7 d is worse in every bucket (sa1 +1.7%) and every day, most on Saturday
  (+1.7%). A 7-day window holds only 2 weekend days for the weekday/weekend table.
- Route 122 moves the opposite way to the total (14 d 340, 21 d 336, 7 d 321). Route 133 improves with the shorter
  lookback (206 → 200 at 7 d; 208 at 21 d). Both are sparse or drifting routes, where a recent table matters more
  than a large one, but each is a single route on one split, so this is no claim.
- Worst routes (21 d): 122 336, 133 208, 106 167, 125 159, 129 132, 111 131, 881 129, 137 126, 113 125, 131 125.
  (7 d): 122 321, 133 200, 106 164, 125 160, 881 134, 137 133, 111 131, 113 130, 129 127, 131 123.

**Conclusion: no new leader. The 14-day lookback production would use for the static tables is the right choice;
21 d adds nothing and 7 d costs 0.9%. The shift split was not run (neither arm is a candidate).**

### Next steps
1. The model-side search is saturated: every family tried in runs 13–17 lands within ±0.3%, except the dwell table (−0.8%)
   and a shorter lookback (+0.9%, worse). Further hourly runs are unlikely to find a ≥3% win over sc_big_dwell. The useful
   next work is hand-off (a serving prototype of sc_big_dwell), and that is the owner's decision.
2. If the search continues without a hand-off, the only untested lever is recency weighting inside the tables (for
   example a 14-day table with the last 3 days weighted up, aimed at drifting routes such as 133 and 122). Expect ±0.3%.
