# Design notes

Running log, one section per component. Updated each milestone.

## M1 components

### config.py
All parameters as frozen dataclasses (`MarketConfig`, `FillConfig`, `StrategyConfig`, `SimConfig`) with validation.
- **Why:** a run is fully described by one printable object.
- **Validation:** negative sigma/A/k/delta/eta, non-positive T or N, `A*dt > 1` raise `ValueError`.
- **Weakness:** `A*dt <= 1` only guarantees p <= 1 at distance 0. Negative distances (aggressive skew) can still hit the cap, which is handled at runtime (see fills).

### midprice.py
Zero-drift arithmetic random walk, `m_{n+1} = m_n + sigma*sqrt(dt)*Z`.
- **Interface:** `MidpriceProcess` has `draw_shocks(rng)`, `path_from_shocks` (batch, `(..., N+1)`) and `stream(shocks)` (tick generator). Shocks are drawn separately from path-building, so batch and live build paths from identical numbers.
- **Bit-exactness:** both add increments left to right from `m_0` (`np.cumsum` is sequential), so the generator equals the batch path exactly (tested).
- **Weakness:** the arithmetic walk allows negative prices. Irrelevant at m0=100, sigma=2, T=1 (sd of m_T is 2), but it is a model property.

### draws.py (not in the original layout, added for the seeding contract)
Session `i` uses `SeedSequence(seed, spawn_key=(i,))`, which is exactly the i-th child of `SeedSequence(seed).spawn(n)` (tested), but can be built lazily for live mode.
- **Draw order:** logically per step `Z, U_bid, U_ask`. Physically, per session: a block of N normals, then an `(N, 2)` block of uniforms. This is a deliberate simplification so batch (vectorised) and live consume the same numbers. The per-session result is identical to step-interleaved ordering in distribution, but it is a different stream.
- **Consequence:** session i never depends on n_sessions or on run mode (tested).

### fills.py
`p = min(1, A*exp(-k*d)*dt)`, fill iff `U < p`. Also exposes where the cap binds.
- **Assumption:** at baseline fills are independent of the future price move (no adverse selection). Adverse selection is now an opt-in layer: `step` takes an `informed` array and `fill_boost` and multiplies p on the picked-off side (see adverse.py). The cap applies to the product, so it can bind on boosted sides.

### strategies.py
`Quoter` ABC with `distances(m, q, t) -> (d_bid, d_ask)`; `InventorySkewQuoter(delta, eta)` with `d_bid = delta + eta*q`, `d_ask = delta - eta*q`.
- eta = 0 is naive. Avellaneda-Stoikov (`eta_t = gamma*sigma^2*(T-t)`) or inventory caps are added as new `Quoter` subclasses; `t` is already passed in.
- **Note:** quotes are not rounded to a tick grid. Every quote distance is floored at `FillConfig.min_dist` (default 0.02) inside `core.step`, so strong skew (eta*|q| >= delta) pins the quote at 0.02 from the mid instead of crossing it; `clamp_steps` counts ticks where the floor bound. `min_dist = -inf` restores the old unbounded behaviour (used by the negative-distance tests).

### core.py
Pure `step(state, m_next, u_bid, u_ask, quoter, fills, dt, t) -> (state, info)` on `(n,)` arrays. It does not mutate its inputs (tested).
- **Order:** quotes from `(m_n, q_n)` -> fills -> inventory/cash -> mid moves -> MTM.
- **Identity:** `Pi_{n+1} - Pi_n = F_b*d_bid + F_a*d_ask + q_{n+1}*(m_{n+1} - m_n)`. Check: the cash change is `-F_b*b + F_a*a`. The revaluation `q_{n+1}m_{n+1} - q_n m_n` splits into `q_{n+1}(m_{n+1} - m_n) + (F_b - F_a)m_n`. The `m_n` terms combine with the cash change into the quote distances. Summing gives `Pi_N = spread_pnl + inventory_pnl` exactly (tested to 1e-9 for eta in {0, 0.1, 0.5}, with the floor binding, and with the floor disabled).
- **Reading it:** spread P&L is what the maker earns for providing liquidity. Inventory P&L is the mark-to-market on whatever is held. With zero drift its expectation is about 0 for the naive maker, so skewing cannot raise spread P&L. It can only reduce inventory variance, and pays for that by quoting worse prices on the side it wants to fill less.
- `simulate_paths` is a thin loop over `step` that records paths (M3's batch engine will reuse it).
- **Cap warning:** a `UserWarning` is issued if the min(1, .) cap binds on more than 0.1% of quote-sides.

### plots.py
Fixed-filename PNGs to `figures/` (overwritten). M1: `single_session.png`. Naive = blue solid, aware = orange dashed, so colour is never the only cue.

## Live dashboard components (Phase A)

### epochs.py
An epoch is a maximal run of sessions with an identical config; statistics (mean, se, per-session Sharpe, worst drawdown, max|q|, detection / false-alarm rates) are computed per epoch from completed sessions only. A config change closes the epoch; the session in flight is recorded as interrupted on the epoch it ran under and counts in no statistic.
- **Weakness:** a thin epoch has a wide se (shown). Epochs with no sessions at all (slider dragging at a boundary) are kept in the log but omitted from the table.

### adverse.py (adverse selection, shared by live and batch)
Hidden informed episodes per session, built by a small state machine from three uniform streams drawn LAST in each session's stream (`u_start, u_dir, u_dur`), so the core `Z` and `U` draws are bit-identical to the pre-adverse streams (tested).
- **Model, precisely.** Quiet -> (V_start < `p_episode` = 0.004/tick) -> LEAD of 10 ticks (book tilt only) -> ACTIVE for `1 + floor(Exp(60))` ticks -> quiet. Direction s = +1/-1 from `V_dir`. While ACTIVE: (a) the mid drifts by `strength * sigma * sqrt(dt) * s` per tick (added by the `MidpriceProcess`, so the path stays a function of the draws), and (b) the picked-off side is hit `fill_boost` = 2.5 times as often (s = +1 -> ask lifted, s = -1 -> bid hit; applied in `core.step`). So an informed fill is followed, on average, by a move against the maker. `informed[n]` is the ACTIVE direction; `tilt[n]` is the LEAD+ACTIVE direction.
- **Single toxicity parameter:** `strength`, in units of the per-tick diffusion scale, so it means the same at any sigma. `strength = 0` disables everything.
- **Default calibration:** 0.12 (`scripts/calibrate_toxicity.py`). Paired against strength 0 on identical draws, it adds ~15% to the variance of a session's net move `E[(m_N - m_0)^2]` and ~0.2% to the realised variance (sum of squared tick changes): realised variance is nearly blind to a persistent drift. Matching +10-20% in the realised-variance sense would need strength ~ 1, which inflates the net-move variance ~11x and makes a 380-tick fall of >= 48 points at sigma 4.86 a ~6% event; at 0.12 none occurred in 7.6M overlapping windows (Gaussian estimate ~1e-10). The 48-point question: driftless it is a ~7.2-sigma event, so it was never a tail draw, it was the old drift.
- **Weaknesses:** one episode at a time; constant strength within an episode; the drift is a known constant; `fill_boost` is a constant independent of strength, so as strength -> 0+ there is informed flow with no edge; the whole mechanism is a stylised stand-in, not calibrated to any market.

### live_sim.py (live = batch, one tick at a time)
`LiveMarketMaker` contains NO accounting of its own. A session is prepared exactly as a batch session (same `draw_session`, `build_regime`, midprice with informed drift via `stream`) and advanced one tick at a time through `core.step` with n = 1. Equivalence with the batch engine is therefore structural, and `tests/test_equivalence.py` checks it session by session (terminal, spread and inventory P&L to 1e-12 relative, fills, max|q| and clamp counts exactly) with adverse selection on, for eta in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2}, with it off, and at other market settings; the large-eta cases assert the 0.02 floor actually bound. Mutation check: dropping the fill boost or the floor in the live path makes the test fail. Live-only pieces: synthetic book, detector, `WideningQuoter`, `auto_eta`.
- **Sessions/epochs:** sessions of N = 200 ticks, session i from `session_rng(seed, i)`; the book uses a separate stream `spawn_key=(i, 1)`. `request()` queues a change; `apply_pending()` applies it at a tick boundary only. Params are constant within a session, so the whole session is prepared at its start. `auto_eta` moves a state variable (`eta_eff`), not the parameter, so it never opens an epoch.
- **Synthetic book (leakage note).** The book is GENERATED FROM THE HIDDEN DIRECTION: depth is tilted by `BOOK_TILT * tilt[n]` plus AR(1) noise. Detection rate therefore reflects THIS BOOK'S LEAKAGE (set by `BOOK_TILT`), not a general adverse-selection signal. Evidence at default toxicity (held-out seeds): book term alone detects 67%, the fill-markout term alone ~1%, and the full detector with `BOOK_TILT = 0` ~1.7%. Treat the detector as a demonstration of mechanics, not as evidence that adverse selection is detectable from real order books.
- **Scoring label.** A tick is labelled "informed" from the start of the lead-in (when the book tilt starts) through the end of the active episode (`tilt != 0`). Previously the 10 lead-in ticks were labelled quiet, so alerts raised there counted as false alarms. Corrected false-alarm rate at the default config on held-out seeds: 1.10% (2.60% under the old labelling); detection rate 74.5%.
- **Known weaknesses:** (1) the detector's early-session fallback reads the true `sigma` parameter (a trader would estimate it); (2) detector constants were chosen a priori but results on dev seeds 0-7 were seen before the drift scale was changed; (3) the first 20 ticks of a session have no volatility estimate.

## Calibration note: tick-level volatility vs session-level displacement risk
Under iid Gaussian increments these are the same thing: the variance of the net move is just the sum of the tick variances, so realised variance (the sum of squared tick changes) and `E[(m_N - m_0)^2]` agree. Once the drift has structure (a persistent, same-signed drift for tens of ticks, as in an informed episode) they separate. A drift of `s*sigma*sqrt(dt)` per tick adds only `s^2` to each tick's second moment, but the drift compounds linearly over an episode of L ticks, so it adds about `(s*L)^2` to the squared net move. At s = 0.12 and episodes of ~60 ticks that is ~0.2% on realised variance but ~15% on net-move variance (and ~11x at s = 1). Inventory risk is driven by how far the price travels while the maker holds inventory, i.e. by the displacement, not by tick-to-tick noise. That is why the default is calibrated on net-move variance (`scripts/calibrate_toxicity.py` prints both, paired against strength 0 on identical draws), and why a realised-variance estimate computed on this generator would wrongly suggest the informed flow is negligible.

## Phase B components

### Paired strategies (live_sim.py, epochs.py)
Every live session runs TWO accounting states through `core.step` on the same mid, the same fill draws and the same informed flags (common random numbers): "aware" = the configured strategy (eta slider, Auto-eta, Auto-widen), "naive" = eta 0 with the same delta, no detector, no widening. Tested against `simulate_paths` for both etas (terminal, spread P&L and max|q| per session). The detector reads the aware strategy's own fills only.
- **Paired statistic:** per session `d_i = aware_i - naive_i` (terminal P&L); per epoch mean(d), se = sd(d)/sqrt(n), n, t. Pairing helps because both strategies share every draw, so sd(d) is well below the two strategies' own spreads (tested). `Epoch.running_paired()` gives the running mean and se after 1..n sessions (checked against brute force).
- **Weakness (optional stopping):** the dashboard shows the running paired mean and se while you watch. Stopping when the band excludes zero is optional stopping and inflates false positives. n per epoch is small and the figure is for intuition; claims come from fixed-n seeded batch sweeps.
- **Weakness:** with eta = 0 and no auto flags the two strategies are identical and the difference is exactly 0 with se 0 (tested); the paired t is then undefined (nan).

### Dashboard (scripts/live_dashboard.py)
`build_dashboard(seed)` builds the figure and returns handles (used by the smoke test); `main()` only adds the animation.
- **B2, scaling:** inventory, MTM P&L, spread P&L and inventory P&L are separate stacked panels, each with its own padded y-limits computed from its own series (tested: independent axes, each covering its data, none blown out by another series). Naive = blue solid, aware = orange dashed, as in `plots.py`.
- **B1, paired view:** bottom-left chart of the current epoch's per-session differences with the running mean and a +-1 se band; the epoch text block shows per-strategy mean +- se, Sharpe, and the paired difference with se, n and t.
- **B3, run-wide panel:** max |q| so far, worst drawdown so far (both strategies), sessions completed, sessions incomplete, current epoch. "Run-wide" = every tick of every session since the run started or was reset, across all epochs, interrupted sessions included (they happened); it never rolls off with the 400-tick chart window (tested against an oracle recomputed from the tick stream, across an interruption and an epoch change). Reset clears it.

### Dashboard widget lifetime (the Reset/Freeze bug)
Matplotlib widgets are kept alive only by the *caller's* reference: the figure holds their axes, not the widget objects. `build_dashboard` returned the sliders and the checkbuttons but not the three `Button`s, so `Reset` and `Freeze` were garbage collected as soon as it returned - still drawn, still clickable, callbacks silently dead. `Pause` survived by accident, because the returned `toggle_pause` closure captured `btn_pause` through `set_paused`; that is exactly the reported symptom (Pause worked, the other two did nothing).
- **Fix:** every widget is returned in `buttons=` / `sliders=` / `chk=` and must stay there.
- **Why the old test missed it:** `test_dashboard_..._freeze` called `d.do_freeze()` directly, which tests the handler but not the wiring. Tests now click through the canvas event path (`MouseEvent(...)._process()` for press + release) and one test asserts no widget is collected after `build_dashboard` returns. Mutation-checked: reverting the fix fails 5 tests (and leaves `pause` passing, reproducing the real symptom).
- **Visibility:** every click goes through an `action()` wrapper that writes to a status line under the paired chart and turns it red with the exception type on failure (tested), so a dead or raising handler can no longer fail silently. Space bar toggles pause.

### repro.py and the Freeze button (B4)
`reproduction_report(epoch)` renders the epoch as Python source: exact float reprs of the config, the seed, the epoch's first session index (`Epoch.start_session`, recorded when the epoch opens; sessions are reproducible without replaying earlier ones because session i draws from `(seed, i)`), the re-run, and an assert that aware / naive / paired-difference statistics equal the recorded ones (`np.allclose`, rtol 1e-9). A script that runs silently has reproduced the epoch; a tampered number fails (tested).
- **Two forms:** "live" drives `LiveMarketMaker(first_session=...)` and always works, including Auto-eta / Auto-widen; "batch" uses `prepare_batch` + `simulate_paths` and exists only when both auto flags are off (the batch engine has no detector or hysteretic quoter). Both reproduce the same epoch exactly (tested).
- **Freeze** pauses the run and prints the report of the current epoch.
- **Limitation:** reproduction assumes the same code version; the live detector/book constants are listed in a comment but there is no version stamp.

## B+2(a) book.py: limit order book with queue position (state model, not yet wired to fills)

Prices live on an **absolute** grid of size `tick`; queue position is only meaningful if a level is a fixed price, so the depth array is stored relative to the mid's anchor while our own order is held by its absolute grid index and does not move when the mid does.

- **State.** `depth[n, side, level]` is *other participants'* resting size; our order is `rest` (absolute grid index), `ahead`, `size`, `live`. `behind = depth[our level] - ahead` is derived, never stored. Invariant `0 <= ahead <= depth[our level]` is asserted every tick of a full session by the tests.
- **Tick order (no lookahead).** 1 requote -> 2 cancels -> 3 adds/replenish -> 4 market orders -> (caller moves the mid) -> 5 reanchor. Cancels are uniform over the queue, so the same fraction leaves `ahead` and our priority improves; adds join the back and leave `ahead` alone. Market orders consume from the touch outward and, within our level, take everything ahead of us, then us, then everything behind.
- **Requote priority.** Posting the same price keeps position; any price change cancels and rejoins at the back (`ahead := depth` at the new level). This holds whether or not the price is currently inside the modelled window - an order does not lose its place because the mid wandered away. `requotes` and `ticks_resting` are counted per side.
- **Window boundary.** The model spans `levels * tick` (default 16 x 0.25 = 4.0) each way. A quote beyond it keeps its price and priority but cannot be filled, since no market order large enough to sweep that far is modelled. This is a modelling boundary, so the fraction of ticks spent outside is reported with every result.
- **Determinism.** All randomness comes from two new `SessionDraws` blocks (`u_queue` (N, 8), `u_depth0` (2, MAX_LEVELS)) appended at the END of the stream with FIXED shapes, so earlier blocks stay bit-identical and changing a book parameter never shifts them (tested). Normal quantiles come from an inline rational approximation (`_ppf`, accuracy tested to 2e-6) rather than a Generator, so batch and live consume the same pre-drawn uniforms.

### Calibrating the defaults to trade at all (an instructive failure)
The first guesses (depth_base 20, mo_size_mean 12) produced **exactly zero fills**. At the default half-spread, delta = 0.75 is 3 ticks, so a market order must clear the cumulative depth of levels 0-2 *plus* our own level - about 122 lots - which has probability `exp(-122/12) ~ 4e-5`. Provisional defaults (depth_base 8, mo_size_mean 45, mo_rate 0.5) put that threshold near 49 lots and give ~0.34 fills/tick/side against 0.227 for `p = A*exp(-k*0.75)*dt`. These are not fitted to any instrument. B+3 (below) settled their status: they are **tuned to fill rate, not calibrated to real books**, and deliberately kept that way, because B+3 showed the two targets cannot both be satisfied.

**This is the substantive point of the queue model:** the Avellaneda-Stoikov intensity assumes a quote 3 ticks off the mid still trades on 23% of ticks, which a queue book only reproduces if it is thin relative to market-order size. The two models can be made to agree on the *average* fill rate, but they disagree on its *composition*.

### Measured behaviour (2000 sessions x 200 ticks, seed 1, delta 0.75)
| eta | fills/tick/side | requotes/tick/side | vs naive | outside window | mean ticks resting |
|---|---|---|---|---|---|
| 0.0 (naive) | 0.341 | 0.286 | 1.00x | 0.0% | 0.61 |
| 0.1 | 0.333 | 0.309 | 1.08x | 1.7% | 0.57 |
| 0.2 | 0.210 | 0.381 | 1.33x | 20.8% | 0.77 |
| 0.3 | 0.107 | 0.418 | 1.46x | 36.1% | 0.99 |
| 0.5 | 0.031 | 0.432 | 1.51x | 48.0% | 1.19 |

- **The requote asymmetry is real but small at the default eta:** the aware strategy requotes 8% more often than naive at eta = 0.1, rising to 46-51% more by eta = 0.3-0.5. Naive already requotes on 29% of ticks purely because the mid wanders across tick boundaries, which is most of the total.
- **Queue position is rarely earned.** We sit on average 65-87% of the way back in our level and rest 0.6-1.2 ticks before the price moves or we are filled. With `sigma*sqrt(dt) = 0.14` against a 0.25 tick, the mid crosses a boundary every few ticks, so a maker who re-derives its quote from the mid every tick cannot build priority. **Expect the stage (c) effect to be a modest drag at eta = 0.1, not a reversal** - but that is a prediction from the state model alone, to be confirmed once fills drive P&L in (b)/(c).
- Fills concentrate at levels 2-4 (73% at eta = 0.1), i.e. around the quoted distance, as intended.

**Weaknesses:** one order per side; continuous sizes rather than integer lots; a single cancel fraction and add amount per side per tick rather than per level, so levels are not idiosyncratically noisy after initialisation; cancels uniform over the queue (real cancels concentrate at the back); no iceberg orders; our order never influences anyone else.

## B+2(b,c,d) queue-position fills in the shared step

`core.step` now takes an optional `QueueConfig`. With `queue=None` it is the exponential model, bit-identical to every earlier result (asserted); with a config it runs the book of B+2(a). Only the fill mechanism differs - quoting, accounting, adverse-selection drift and the P&L identity are one code path.

### (b) Partial fills and size-weighted spread capture
Fills are now SIZES (floats), not indicators, so `q`, `bid_fills`/`ask_fills` and `max_abs_q` are floats throughout. Two consequences:
- **Spread capture must use the EXECUTED price.** Quotes are rounded to the grid, so the price actually resting is at least `d` from the mid but not exactly `d`. The increment is `F_bid*(m_n - bid) + F_ask*(ask - m_n)`, weighted by filled size. A test rebuilds `spread_pnl` from executed prices and mids and matches the counter to 1e-10, and asserts the requested-distance version does NOT match.
- **The identity needs no unit-size assumption.** `Pi_{n+1} - Pi_n = F_b*(m_n - b) + F_a*(a - m_n) + q_{n+1}*(m_{n+1} - m_n)` holds for any prices and any fractional sizes; verified to 1e-9 on a run where 2.9% of fills are partial.

### Composition: matching the mean fill rate is the floor, not the result
1500 sessions, eta = 0.1, default toxicity. Rates were tuned to the same order of magnitude; the process still differs sharply.

| model | fills/t/side | partial | burstiness | picked off | rate given informed | rate given quiet |
|---|---|---|---|---|---|---|
| exponential | 0.253 | 0.0% | 0.020 | 0.063 | 0.370 | 0.235 |
| queue | 0.365 | 2.9% | 0.110 | 0.263 | 0.530 | 0.338 |

- **Burstiness** (lag-1 autocorrelation of "traded this tick") is 5x higher under the queue model: one market order sweeping several levels fills us now and leaves us at the back afterwards, so fills clump. Exponential fills are near-independent by construction.
- **Adverse selection bites 4x harder** under the queue model (`picked off` = `E[signed fill x informed direction]`, 0.263 vs 0.063). Informed flow arrives as *larger market orders on one side*, which both reaches deeper into the queue and is exactly the flow we do not want to meet. The exponential model's `fill_boost` raises a probability but cannot make the fill bigger or deeper.
- This is the substantive difference: **the exponential model understates how concentrated and how toxic fills are**, even when the average fill rate agrees.

### (c) The cost of losing queue priority - both predictions were wrong
Measured by a paired counterfactual: `QueueConfig.keep_priority` lets an order carry its queue place when its price moves (no venue allows this), so running it against the realistic setting on identical draws isolates the cost. 1500 sessions.

| eta | P&L (priority costly) | P&L (requoting free) | cost | se | t |
|---|---|---|---|---|---|
| 0.0 naive | 116.13 | 121.48 | **-5.35** | 0.12 | -44 |
| 0.1 | 128.81 | 133.26 | **-4.45** | 0.09 | -52 |
| 0.2 | 127.77 | 132.37 | -4.60 | 0.10 | -48 |
| 0.3 | 125.42 | 130.24 | -4.82 | 0.10 | -47 |

Requoting costs 3.5-4.5% of P&L and is overwhelmingly significant - but it is **flat in eta, and largest for the naive strategy**. The differential, which is what the research question needs, is `+0.90 +- 0.15` in the aware strategy's favour at eta = 0.1: skewing is very slightly *cheaper* in priority terms, not more expensive.
- The original hypothesis (skewing requotes more, so it should pay more) fails because naive already requotes on 27% of ticks from mid-wander alone; the skew term adds only ~2 percentage points.
- My own weaker prediction ("a modest drag at eta = 0.1") is also wrong in sign: there is no differential drag.

### When does priority matter? Two different questions
Sweeping `sigma*sqrt(dt)/tick`, how far the mid moves per tick in grid units:

| sigma*sqrt(dt)/tick | ticks per requote | priority cost | as % of P&L |
|---|---|---|---|
| 0.071 | 8.65 | -1.79 | -1.2% |
| 0.283 | 5.38 | -2.97 | -2.1% |
| 0.566 | 3.40 | -4.45 | -3.3% |
| 1.131 | 2.23 | -6.70 | -5.6% |
| 2.263 | 1.68 | -8.56 | -9.5% |

The prediction was that priority should matter most when the mid is SLOW, because a quote survives long enough to climb a queue. The survival half is right - 8.65 ticks per requote at the slow end against 1.68 at the fast end - but the **P&L cost moves the other way**. The cost is `(how often you requote) x (value of a queue place)`, and the first factor dominates: when the mid is fast you are permanently at the back of the queue, so the counterfactual of free requoting is worth much more.

So the two questions come apart, and the useful statement is: *a slow mid relative to the tick is where priority can be **earned**; a fast one is where failing to keep it **costs** most.* Both scale with the same ratio, in opposite directions.

### (d) Equivalence
`tests/test_equivalence.py` runs the live tick loop against the vectorised batch engine session by session for eta in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2}, with the queue model on, with it off, with adverse selection on and off, and at a second market setting. Terminal / spread / inventory P&L match to 1e-12 relative, fills and max|q| to 1e-12, clamp counts exactly.
- **Window coverage is a non-issue after all.** Stage (a) reported 20-48% of quotes outside the modelled window at high eta; that was an artifact of the diagnostic script quoting *unclamped* negative distances (a crossed quote reads as "outside"). `core.step` floors distances at `min_dist`, and in the real engine the outside fraction is 0.00% at eta <= 0.1 and 0.93% at eta = 1.0, because when the skew works inventory stays small. No sweep tail needs excluding.
- **Mutation-checked**, and this exposed two things worth recording. (1) Copying the project to a scratch directory and mutating there proves nothing: `mmsim` is installed as an **editable** package, so the copied tests still import the original `src`. Package-level mutation checks must be done in place and reverted. (2) Once done properly, a mutation of the live *naive* quoter passed - `assert_equivalent` only compared the aware leg. Fixed by checking `naive_terminal_pnl`/`naive_spread_pnl`/`naive_max_abs_q` against a batch run at eta = 0; all three live-only mutations (off-by-one in `u_queue`, dropped `tilt`, perturbed naive delta) now fail 9 tests each.

**Weaknesses:** the `keep_priority` counterfactual is not a market feature and exists only to decompose the cost; its first implementation was wrong (it pinned the order at its old price instead of moving it for free), which showed up as a positive "cost" - the sign of a counterfactual is worth checking before reading anything into its magnitude.

## Headline comparison so far (queue model, default toxicity 0.12)

Paired over 5000 sessions on identical draws, seed 2024, delta 0.75, aware eta 0.1:

| fill model | aware mean (sd) | naive mean (sd) | paired diff | se | t | Sharpe aware / naive | Sharpe gap |
|---|---|---|---|---|---|---|---|
| queue | 128.73 (26.37) | 115.83 (47.20) | **+12.90** | 0.68 | 19.0 | 4.881 / 2.454 | +2.427 |
| exponential | 71.21 (9.49) | 69.31 (24.15) | **+1.90** | 0.33 | 5.8 | 7.502 / 2.870 | +4.631 |

The aware strategy wins on both models and the result is not marginal. Two cautions before reading more into it: (1) this is a single toxicity setting, which C1 sweeps; (2) the two models rank *differently* depending on the statistic - the queue model shows a far larger P&L gap but a *smaller* Sharpe gap, because the aware strategy's own dispersion is much higher under it (sd 26.4 vs 9.5). Reporting only one of the two would be misleading.

## Prediction registered before C1 (so it can be scored, not rationalised)

**Claim.** The Sharpe gap (aware minus naive) will rise MORE STEEPLY with toxicity under the queue model than under the exponential one.

**Mechanism.** The queue model makes informed flow about 4x more damaging at matched mean fill rates (`picked off` 0.263 vs 0.063). It does this structurally: informed flow arrives as *larger, one-sided market orders that reach deeper into the book*, so a toxic episode both fills us more often and fills us in bigger size on the wrong side. The exponential model can only raise a fill probability; `fill_boost` cannot make a fill bigger or deeper, so its adverse selection saturates. If inventory skew is what protects against adverse selection, its advantage should therefore grow faster with toxicity wherever adverse selection is itself more damaging.

**What would falsify it.** Slopes equal within bootstrap CIs, or the exponential slope steeper. A plausible failure: at default toxicity the exponential model already shows the LARGER Sharpe gap (+4.63 vs +2.43), so the queue model starts from a lower base and may simply have more room above - or may be dominated by its own higher variance, which enters the Sharpe denominator and could flatten the gap regardless of how toxic the flow gets.

**Scored in the C1 section below after the sweep runs.**

## Process lessons (recorded because each one let a real defect through)

### Editable installs make copy-and-mutate verification worthless
`mmsim` is installed with `pip install -e`, so `import mmsim` resolves to `src/mmsim` through a `.pth` file **no matter which directory the tests run from**. Copying the project to a scratch directory, mutating the copy and running its tests therefore exercises the ORIGINAL package: the mutation is never loaded and every test passes, which reads exactly like "the tests do not catch this". Two mutation checks were silently meaningless before this was noticed (confirmed by printing `mmsim.__file__`).
- **Do instead:** mutate the package file in place, run, restore from a backup copy, and verify the restore (`grep -c` for the mutant text). Only file-path-loaded code, such as `scripts/live_dashboard.py` in the B+1 check, is safe to mutate in a copy.
- **Why it matters:** done properly, the very first package-level mutation found that `assert_equivalent` compared only the AWARE leg. The live simulator's paired naive strategy was covered by no equivalence test at all, so any naive-side bug would have passed the entire suite while silently corrupting every naive-vs-aware comparison - that is, the project's actual research question.

### Check a counterfactual's SIGN before reading its magnitude
The first `keep_priority` implementation kept the order at its OLD price instead of moving it to the new one for free, so the "requoting is free" arm quoted a stale price and traded less. The measured priority cost came out POSITIVE - losing priority appeared to help - which is not a subtle error but is easy to write up as a surprising finding. A counterfactual whose sign contradicts the mechanism is a bug until proven otherwise; check the sign first, and only then the size.

### Draw only what a run will use; an "always drawn, fixed order" seeding contract does not require "always drawn, fixed order, EAGERLY"
`draws.py`'s seeding contract (every block appended at the end, in fixed order, so earlier blocks never shift) was implemented by drawing EVERY block unconditionally, including the queue-model's `u_queue`/`u_depth0`, even for runs that never pass `queue=` to `core.step`. At n sessions x N=200 ticks this is `n*200*8*8` bytes for `u_queue` alone (12.8 MB per 1,000 sessions) allocated and then thrown away - a 200,000-session exponential-only diagnostic (C4's dominance check) allocated several GB it never read and had to be killed. **Fixed properly, not worked around**: `draw_session`/`draw_batch`/`prepare_batch` now take `want_queue` (default True, so untouched call sites keep their exact prior behaviour and seeding); `False` skips drawing the queue blocks entirely - they come back as `None` - rather than drawing-and-discarding them. This is safe because each session owns an INDEPENDENT generator (`session_rng`, via `SeedSequence` spawn), so how much of it a given caller chooses to consume has no effect on any other session or on the non-queue blocks of the SAME session, which stay bit-identical whether or not the queue blocks are drawn afterward (tested in `tests/test_draws.py`, including a direct memory-delta measurement via `tracemalloc` that isolates the queue contribution from the six always-drawn fields' own, much smaller, cost). Every exponential-only call site in the project (`sweep_sensitivity.py`, `validate_closed_form.py`, the exponential legs of `power_analysis.py`/`compare_fill_models.py`) now passes `want_queue=False` explicitly. The original fix in the field was to just kill the hung process and cap `n`, which worked around the symptom; the actual fix removes the allocation at its source.

## C1: adverse-selection sweep (2-D), and the registered prediction scored

### Why 2-D: the 1-D sweep's x-axis was confounded
The first version of C1 (`scripts/sweep_toxicity.py`) swept `AdverseConfig.strength` alone. Reading its results surfaced a confound: `strength = 0` is not "a little less toxic" - `build_regime` short-circuits on `strength == 0` and emits NO episodes at all, so `QueueConfig.informed_mo_mult` (the one-sided market-order flow boost) has nothing to act on and disappears with it. The first grid point is a different regime from the rest, not a limit of the others. Holding the flow boost at 1.0 while varying strength showed the Sharpe gap barely moves (4.54 to 4.26 across strength 0.05-0.50), while holding strength fixed and varying the boost moved it by several points - i.e. almost the entire "toxicity" effect in the 1-D sweep was the flow boost switching on, not the drift increasing. That result is kept here only as the reason the 1-D sweep was superseded; the 2-D grid below is the one to read.

### Design
`scripts/sweep_toxicity_2d.py`, figure `figures/toxicity_sweep_2d.png`. 10,000 paired sessions per point, seed 2024, eta 0.1 vs 0.0, paired bootstrap (2000 resamples, one index draw applied to both strategies, so the pairing survives resampling). Two independent axes:
- **rows** `informed_mo_mult` in {1.0, 1.2, 1.5, 1.8, 2.0} - the one-sided flow boost, including 1.0 (boost OFF, drift only) as an explicit row so "flow does the work" is visible as a flat line rather than inferred from a surface.
- **columns** `strength` in {0.05, 0.10, 0.20, 0.30, 0.50} - the informed price drift.
- `strength = 0` is reported once, separately (not as a grid row), since with no episodes `informed_mo_mult` cannot have an effect there.

### An arrival-probability saturation, found while sizing the grid
The first grid attempt used `informed_mo_mult` in {1.0, 1.5, 2.0, 2.5, 3.5} and produced three IDENTICAL rows (2.0, 2.5, 3.5). Cause: informed arrival is drawn as `u < min(1, mo_rate * mo_mult)` with `u ~ Uniform[0, 1)`; at the default `mo_rate = 0.5` this reaches 1 at `mo_mult = 2.0`, and because `u` is strictly less than 1, ANY threshold at or above 1 makes arrival certain on every tick - larger multipliers cannot do more. **The C1 1-D sweep's default (`informed_mo_mult = 2.5`) was already past this point**, so its queue-model toxicity axis was, in effect, binary (episodes on/off) once past the first column. Note for anyone mutation-testing this: the `np.minimum(1.0, ...)` call is itself a no-op given the uniform-threshold comparison - removing it changes nothing, because `u < p` for any `p >= 1` already equals `u < 1`. The saturation is a property of Bernoulli-via-uniform-threshold sampling, not a guardable clamp; `tests/test_toxicity_2d.py` protects the empirical claim (identical results at and past the threshold, different results below it) rather than a line of code. The grid below is redesigned to stay at or below the threshold so every row is genuinely different; past it, only fill SIZE (not frequency) could add more toxicity, and `mo_size_mean` does not depend on the informed state - a real limitation to address in B+3, not a tuning choice.

### Results

**Paired P&L gap** (aware - naive terminal P&L):

| mo_mult | strength 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| 1.0 (flow off) | -0.35 | 0.71 | 2.81 | 4.77 | 8.80 |
| 1.2 | 1.17 | 2.72 | 5.78 | 8.73 | 14.75 |
| 1.5 | 3.17 | 5.19 | 9.22 | 13.15 | 21.07 |
| 1.8 | 5.63 | 8.11 | 13.09 | 17.90 | 27.75 |
| 2.0 (saturates) | 8.34 | 11.32 | 17.23 | 23.00 | 34.80 |

**Sharpe gap** (aware - naive Sharpe):

| mo_mult | strength 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| 1.0 (flow off) | 4.397 | 4.655 | 5.004 | 4.969 | 4.307 |
| 1.2 | 4.007 | 4.595 | 5.495 | 5.545 | 4.350 |
| 1.5 | 2.892 | 3.670 | 5.229 | 5.744 | 4.286 |
| 1.8 | 1.927 | 2.680 | 4.473 | 5.510 | 4.141 |
| 2.0 (saturates) | 1.366 | 2.017 | 3.750 | 5.082 | 4.025 |

`strength = 0` (no episodes, `informed_mo_mult` irrelevant): P&L gap **-3.45** [-3.75, -3.15], Sharpe gap +3.691 [3.559, 3.817] - naive beats aware (see the validation result below).

### The main hypothesis holds, cleanly
Every row of the P&L-gap table rises monotonically in strength, and higher `informed_mo_mult` lifts the whole curve (range at strength 0.05-0.50: +9.15 with flow off vs +26.46 at the saturation point). **Drift and flow are both genuinely toxic and their effects compose roughly additively in P&L terms** - there is no row where adding drift fails to help, and no column where adding flow fails to help.

### The registered prediction: still half right, now with the mechanism isolated
**Scored: NOT CONFIRMED on the stated metric, same as the 1-D result, but now attributable.** The Sharpe-gap surface is not monotonic in strength once flow is present: every row with `informed_mo_mult > 1.0` rises then FALLS (peaking around strength 0.2-0.3), while the flow-off row (1.0) is nearly flat (range +0.70 across the whole strength axis vs +3.72 at the saturation point). **The turnover is a flow effect, not a drift effect** - this is the 2-D grid's answer to where the 1-D sweep's non-monotonicity came from.

**The mechanism remains confirmed on the P&L gap**, which responds to both drift and flow with no turnover anywhere in the grid. The explanation from the 1-D writeup still holds: the queue model's bigger, burstier, one-sided fills inflate the variance of BOTH strategies' outcomes, and that inflation scales with flow strength, so the Sharpe denominator eventually grows faster than the numerator. **Separating a mechanism result from a metric artifact here is more informative than a clean confirmation would have been**: the P&L gap says the queue model's adverse selection is unambiguously more damaging and skewing is unambiguously more valuable against it; the Sharpe gap says that, measured in volatility-adjusted terms, the advantage can appear to shrink at high toxicity purely because toxicity makes outcomes noisier for everyone. Reporting either statistic alone would have been misleading in a different direction.

### Validation result, carried forward to C4 - and CORRECTED there
**At zero toxicity, naive beats aware on MEAN P&L on both fill models** (queue: P&L gap -3.45 [-3.75, -3.15]; exponential: -1.64 [-1.87, -1.41], both CIs excluding zero). This was originally read here as "the textbook Avellaneda-Stoikov inventory/spread trade-off: skewing buys protection nobody needs." **C4 found that framing incomplete and corrects it**: AS's reason to skew is CARA risk aversion over terminal wealth *variance*, not adverse selection - the AS model predicts skewing is utility-improving whenever risk aversion (gamma) is positive, with or without any informed flow at all. Measured on the right objective (certainty-equivalent utility, or even just Sharpe), **aware wins at zero toxicity on both fill models** - see "C4, Q1" below for the full re-examination and the corrected statement. The mean-P&L finding above is still true and still worth knowing (skewing has a real dollar cost even when it is utility-improving), it is just not, by itself, the AS validation it was first described as.

## C4: validation against the Avellaneda-Stoikov (2008) closed form

Regime: exponential fill model (queue OFF), zero toxicity, zero drift, no frictions - the regime the AS closed form describes. `scripts/validate_closed_form.py`, figure `figures/closed_form_validation.png`, tests `tests/test_closed_form.py`. New: `strategies.AvellanedaStoikovQuoter` (exact time-varying policy) and pure functions `as_eta`/`as_half_spread`; `mmsim/utility.py` (CARA certainty equivalent, this project's only use of an objective other than Sharpe/mean P&L, used here because it is the objective AS's derivation actually optimises).

### What is EXACT (verified bit-for-bit against the simulator's own mechanics, no continuum limit, no formula taken on faith from memory)
1. **Myopic optimal half-spread, delta\* = 1/k.** Independent of AS - pure single-variable calculus: `R(d) = A*dt*d*exp(-k*d)` is maximised at `d = 1/k`, for any A, sigma, T. Sweeping delta (eta=0, zero toxicity, n=10,000) finds the simulated E[terminal P&L] argmax at delta = 0.6667 = 1/1.5 EXACTLY, and the exact formula `N*2*p(delta)*delta` matches the simulated mean at every grid point to within bootstrap noise (gaps of 0.03-0.24 against means of 40-69, i.e. well under 1%). The project default delta=0.75 sits 12.5% above this risk-neutral optimum - a design choice that predates this validation, not explained by anything in this zero-toxicity/eta=0 regime (the AS risk-aversion term that would justify a wider spread is exactly zero here).
2. **eta=0 terminal inventory variance, Var(q_N) = N\*2p(1-p).** Bid and ask fills are iid Bernoulli when eta=0 (quotes never move), so this is exact at EVERY dt, not an asymptotic approximation - confirmed by sweeping N from 50 to 800 (dt from 0.02 to 0.00125) and finding the simulated/exact ratio stays at 0.987-1.015 throughout, with no trend. Skewness and excess kurtosis fall toward 0 as N grows (0.044/-0.005 at N=50 vs 0.025/0.040 at N=800) - this part IS the CLT (q_N is a finite sum of iid {-1,0,+1} steps, Normal only as N -> infinity) and is flagged as such, distinct from the exact variance identity.
3. **The `AvellanedaStoikovQuoter` reproduces its own formula** at every (t, q) sampled (tests), and the realised skew (d_bid - d_ask) measurably shrinks over a session for fixed inventory, confirming the time-decay is wired through the `Quoter.distances(m, q, t)` interface correctly, not just correct in isolation.

### gamma: a stated calibration CHOICE, not an independently known number
This project has no risk-aversion parameter anywhere else. Convention used: eta_t AVERAGED over the session (= its value at the midpoint t=T/2, since eta_t is linear in t) is set equal to the project's existing default constant eta=0.10. This gives **gamma = eta_default / (sigma^2 \* T/2) = 0.10 / (4 \* 0.5) = 0.05**, and the implied half-spread at the midpoint is delta_bar=0.7058 (default delta=0.75, 6.3% above) while the half-spread AT THE START of the session is delta_0=0.7558 (0.8% below the default - closer, since the default roughly resembles the policy's value near the start of a session more than its time-average). This is a reverse-engineering of the project's OWN prior choices against the formula, not a literature value for gamma, and is reported as exactly that.

### HEADLINE RESULT - Utility dominance: an apparent violation, investigated, and NOT attributed to a bug
A grid search over constant (delta, eta) for CE utility (gamma=0.05, BACK-SOLVED from the project's own default eta, not a literature value - see above, n=10,000) finds argmax at (0.7, 0.05), CE=66.448, which BEATS the exact time-varying AS policy's CE=65.772 by 0.676 - an apparent violation, since the exact time-varying policy should not be dominated by a constant one if it really is the HJB-optimal control. This is reported as a headline result of C4, not a footnote: investigated rather than asserted either way:
- **Not sampling noise or an overfit grid-argmax**: re-running the exact policy against that SAME (0.7, 0.05) point at n=10,000 / 30,000 / 60,000 gives a STABLE gap of -0.676 / -0.704 / -0.720 - if it were noise or multiple-comparisons bias in the 36-point grid search, the gap would shrink toward zero as n grew; instead it is flat to slightly growing.
- **Not the min_dist quote floor**: clamp fraction is negligible (<0.01% of ticks) for the exact policy, the default, and the grid-best point alike.
- **Most likely explanation**: the widely-cited AS(2008) closed-form spread formula is ITSELF a known approximation to the full HJB solution, not the exact optimal control - Gueant, Lehalle & Fernandez-Tapia (2012) derive the exact solution via a linear ODE system and show the AS closed form approximates it. A small, stable shortfall against a well-chosen constant policy is CONSISTENT WITH that property of the formula - this is NOT claimed to DEMONSTRATE it, since this simulator's own discrete-time mechanics are a further, separate layer of approximation on top of AS's own continuous-time approximation, and the two have not been disentangled here. Reported as a genuine, unresolved finding (per the standing instruction that an unexplained gap is a finding), not smoothed over or mis-attributed to a simulator defect.
- **dt-convergence of the gap, pushed to N=3200 (a 64x range of dt, from 0.02 down to 0.00031) and still INCONCLUSIVE**: the exact-minus-default-constant utility gain is -0.289, +0.302, +0.298, +0.222, +0.188, +0.218, +0.148 at N = 50, 100, 200, 400, 800, 1600, 3200. The three finest-dt successive changes (-0.035, +0.030, -0.069) do NOT shrink monotonically, so no convergence is demonstrated even at a 64x refinement. **The likely cause is NOT the cost of larger N** (cheap, linear, seconds per point) **but Monte Carlo noise**: each N here redraws an entirely INDEPENDENT sample (same seed, but a different N consumes a different number of normal draws, so price paths are uncorrelated across rows rather than a shared-path refinement), and the oscillation is consistent with that sampling noise being comparable in size, at n_sessions=10,000, to the dt-trend being sought. The lever that would actually resolve this is more SESSIONS per point (to shrink the noise), not more ticks per session - left open for a future run rather than guessed at.

### C4, Q1: the zero-toxicity finding, corrected
Recomputed at matched n=10,000 and seed, exponential model, zero toxicity:

| metric | naive | aware | gap (aware - naive) |
|---|---|---|---|
| mean P&L | 68.324 | 66.681 | **-1.643** (se 0.111) |
| Sharpe | 5.063 | 9.538 | **+4.475** |
| CE utility (gamma=0.05, back-solved - see above) | 63.252 | 65.474 | **+2.222** |

Mean P&L favours naive (skewing gives up some spread capture); Sharpe and CE utility BOTH favour aware, because skewing reduces the variance of terminal wealth - exactly the AS mechanism, which needs no adverse selection at all to operate. **The corrected statement is: aware wins on risk-adjusted/utility terms even at zero toxicity, confirming AS's risk-aversion rationale for skewing independent of toxicity; C1 then shows that MORE toxicity makes aware win on raw mean P&L too.** The original framing ("naive beats aware ... the AS trade-off") used the one statistic (mean P&L) that AS's own theory does not optimise, and is amended above rather than left standing.

### C4, Q2: exact vs convergence vs not independently verified
**Exact** (sections 1-2 above): myopic delta\*=1/k; eta=0 Var(q_N)=N\*2p(1-p); the quoter's own wiring.
**Convergence evidence** (asymptotic claims, tested across dt rather than assumed at one dt): inventory shape approaching Gaussian as N grows; the utility-dominance gap's behaviour across dt (inconclusive, reported as such, not claimed to have converged).
**Not independently verified from a remembered formula, explicitly**: a closed-form expression for terminal wealth or inventory variance when eta != 0, or the full AS value function - no such formula is asserted from memory anywhere in this project. gamma itself is a stated back-solved convention, not a sourced number.

### Confirmed by an independent route: the sensitivity sweep (below)
The sigma leg of the five-parameter sensitivity sweep finds the SAME mean-P&L-vs-Sharpe divergence that motivated the Q1 correction, varying sigma instead of toxicity: at low sigma (0.5-1.0) the paired P&L gap is NEGATIVE (-1.70, -0.50 - skewing costs money when there is little volatility risk to protect against) while the Sharpe gap stays positive throughout the ENTIRE sigma grid (0.649 at sigma=0.5 up to 5.376 at sigma=4). Two independent sweeps (toxicity, and now sigma), on two different axes, land on the same structural finding: Sharpe/utility favours skewing far more broadly than mean P&L does, consistent with AS's risk-aversion-only rationale.

## Mean P&L vs Sharpe/utility: a methodological result in its own right

This disagreement has now occurred three times, independently, on three different axes, and is recorded here as a named result rather than left implicit in each section that hit it:

1. **C1's registered prediction** (queue-model toxicity sweep): the Sharpe gap and the paired P&L gap ranked the two fill models OPPOSITELY as toxicity rose - P&L gap grew faster under the queue model (mechanism confirmed), Sharpe gap grew slower and turned over (because the same mechanism inflated variance for both strategies at once).
2. **C4's zero-toxicity finding**: mean P&L favours naive; Sharpe and certainty-equivalent utility both favour aware, at the SAME point. Reading only mean P&L produces the wrong conclusion about whether AS's theory is confirmed.
3. **The sensitivity sweep's sigma axis** (below): mean P&L gap is negative at low sigma, positive at high sigma; the Sharpe gap is positive across the ENTIRE sigma range tested, never crossing zero.

**Why**: in this model the inventory-aware strategy's advantage is substantially a VARIANCE advantage (it reduces the dispersion of terminal wealth), not purely a mean advantage. Sharpe and CE utility are built to reward variance reduction directly; mean P&L is blind to it. Any single-statistic comparison can therefore invert its own conclusion depending on which statistic was chosen. The sigma axis reproducing the same divergence that first appeared on a completely different sweep (toxicity) is the evidence this is a structural property of the model, not an artifact of one sweep's specific setup.

**Practical rule adopted from here on**: report mean P&L AND a variance-aware statistic (Sharpe or CE utility) together for every paired comparison in this project, never one alone, and state explicitly which statistic a given claim is being made on.

## Sensitivity sweep (sigma, A, k, delta, eta)

`scripts/sweep_sensitivity.py`, figure `figures/sensitivity_sweep.png`. Exponential model, DEFAULT toxicity (0.12, the project's normal operating point - not C4's zero-toxicity regime), n=5000 paired sessions per point, 1000-resample paired bootstrap, aware eta=0.10 except on the eta sweep itself (where naive is always eta=0 and the swept value is "aware").

**The Sharpe gap is positive across the ENTIRE grid for all five parameters** - never once crosses zero, from sigma 0.5 to 8, A 40 to 280, k 0.5 to 4, delta 0.3 to 1.5, eta 0.05 to 0.5. This is a strong, independent robustness result: the Sharpe/utility advantage of skewing is not a knife-edge finding tied to the project's specific defaults.

**The paired P&L gap flips sign on two of the five parameters:**
- **sigma**: negative below sigma~1 (-1.70 at 0.5, -0.50 at 1.0), positive from sigma=2 up (as high as +16.32 at sigma=8). At low volatility there is little inventory risk for skewing to protect against, so it purely costs spread capture; the effect reverses once volatility (and the toxicity drift, which scales with sigma) makes inventory risk material. This is the same mean-P&L-vs-Sharpe divergence documented in C4, Q1, found independently via a different parameter.
- **eta**: positive and peaking around eta=0.05-0.10 (+2.26, +1.90), crossing to negative by eta=0.3 (-1.44) and sharply negative by eta=0.5 (-5.44), while the Sharpe gap stays solidly positive throughout (4.09-4.63). There is a mean-P&L-maximising eta near the project's OWN default of 0.10 at this toxicity level - over-skewing gives up more spread capture than it saves in avoided adverse selection, even though it continues to reduce variance (and so continues to help Sharpe) well past that point.

A, k and delta show no sign change over the grids tested: more aggressive informed-flow-independent fill parameters (higher A, lower k) and tighter delta all increase both the mean-P&L and the Sharpe advantage of skewing monotonically over the ranges swept.

Caveat: the cap `min(1, A*exp(-k*d)*dt)` binds on a non-negligible fraction of quote-sides at the extremes of the sigma and A grids (up to 6.82% at sigma=8), which the project's own `CAP_WARN_FRACTION` convention (0.1%) flags as making results "sensitive to the cap" - the DIRECTION of these extreme-grid-point results should be trusted more than their exact magnitude.

## C8: statistical power

`scripts/power_analysis.py`. For six headline paired comparisons reported across this project, RECOMPUTED fresh at the exact n originally used (not pulled from memory of earlier printed numbers), required n for 80% power at 95% confidence via the standard paired-difference formula `n_req = ((z_a/2 + z_b)^2 * sd_d^2) / mean_d^2`:

| comparison | n required | n used | margin |
|---|---|---|---|
| C1 2D, queue, default toxicity (0.12), eta 0.1 vs 0 | 108 | 5,000 | 46.2x |
| C1 2D, exponential, default toxicity (0.12), eta 0.1 vs 0 | 1,185 | 5,000 | 4.2x |
| C1 2D, queue, zero toxicity, eta 0.1 vs 0 | 155 | 10,000 | 64.6x |
| C1 2D, queue, strength=0.5 (mo_mult saturated), eta 0.1 vs 0 | 67 | 10,000 | 149.5x |
| B+2(c) priority-cost differential, eta=0.1 | 273 | 1,500 | 5.5x |
| C4 Q1, exponential, zero toxicity, mean P&L gap | 358 | 10,000 | 27.9x |

Every headline comparison in this project used more sessions than its 80%-power requirement, by a margin of at least 4.2x. The exponential-model comparisons need the most sessions of the batch (the exponential model's smaller fill-outcome variance, ironically, does not translate to needing FEWER sessions here because the underlying EFFECT size at default toxicity is also smaller on that model - see C1's P&L-gap table). No result in this project is currently underpowered at the n it was reported with.

## C2: markout-based fill quality - the composition finding, confirmed fill-by-fill

`scripts/markout_analysis.py`, figure `figures/markout_analysis.png`, tests `tests/test_markout.py`. The natural companion to B+2(b)'s composition finding (queue fills are ~4x more "picked off" than exponential fills at a matched mean rate) - markout is how a real desk actually measures this, fill by fill, rather than via the hidden ground truth the detector tries to infer.

### Decomposition (found necessary while interpreting an apparent anomaly, not planned up front)
`markout_h = distance_captured + impact_h`, both signed `side * (... - ...)`, side=+1 for a bid fill (bought), -1 for an ask fill (sold), positive=good for us:
- `distance_captured` (h-independent): the static compensation charged at fill time, `side*(fill_price - mid)`. Constant (=delta) for naive; WIDENS on whichever side is under pressure for aware, since `eta*|q|` grows exactly there - this is the skew's entire purpose.
- `impact_h`: the PURE subsequent price move, `side*(mid[t+h]-mid[t])`, excluding any pricing choice entirely - the genuine adverse-selection signal, a property of the FLOW, not of the quoting strategy.
- `markout_h`: the desk-standard, fill-price-referenced P&L markout - was this trade, net, a good one.

### Why the decomposition was needed: an apparent sign flip, investigated and explained
The first run showed something that looked like a bug: under the queue model, AWARE's fill-price markout on INFORMED fills was net POSITIVE and even exceeded its markout on QUIET fills (the opposite of every other strategy/model combination). Before reporting this either as a finding or dismissing it, it was checked for stability (stable from n=500 to n=8000, ruling out noise) and decomposed by side and informed direction (confirming the boosted side's markout behaves as the OPPOSITE of the naive "boosted=bad" intuition for AWARE specifically, while matching it for NAIVE). The decomposition resolves it cleanly: `impact_h` (pure toxicity) IS negative on informed fills for every combination tested - the flow really is adverse, independent of pricing. But AWARE's `distance_captured` on informed fills is far larger than its quiet-fill distance (the skew widening on the side under pressure), and at the project's default toxicity (0.12) this OVER-COMPENSATES for the impact, so the fill-price `markout_h` nets out positive. **This is the skew mechanism working as intended, not an absence of adverse selection** - reporting `markout_h` alone would have made it look like no adverse selection occurred; `impact_h` alongside it shows the flow was genuinely toxic and shows exactly how much the skew over/under-charged for it.

### Results at h=20 (n=5000, default toxicity 0.12)
| model | strategy | dist captured (informed) | pure impact (informed) | fill-price markout (informed) |
|---|---|---|---|---|
| exponential | naive | 0.7500 | -0.1285 | 0.6215 |
| exponential | aware | 0.7284 | -0.0263 | 0.7021 |
| queue | naive | 0.8752 | -0.1793 | 0.6959 |
| queue | aware | 1.2764 | -0.0731 | 1.2033 |

- **Composition finding confirmed fill-by-fill**: the queue model's informed-fill impact is more negative than the exponential model's at matched strategy (naive: -0.1793 vs -0.1285; aware: -0.0731 vs -0.0263) at every horizon tested (5, 10, 20, 50 ticks), via a metric a real desk would actually compute - independent confirmation of B+2's "queue fills are ~4x more picked off" via the hidden-state detector.
- **A second protection channel, found while reading the numbers, not predicted in advance**: AWARE's pure impact is ALSO less negative than naive's at every horizon and on both models (exponential: -0.0263 vs -0.1285 at h=20; queue: -0.0731 vs -0.1793) - not just priced better (distance_captured), but genuinely LESS TOXIC on the fills that occur. This is a composition/selection effect: as inventory builds during an informed episode, the skew moves the quote away from the dangerous side, so the surviving fills on that side are increasingly drawn from the EARLY, less-severe part of episodes, before the price has moved far - the worst exposures are avoided rather than merely priced for. Skewing therefore protects through two channels: pricing the risk it does take (distance_captured) AND avoiding the worst of it altogether (impact reduction via fill-composition).
- **The exponential model understates the pricing-side protection**: its distance_captured barely differs between informed (0.7284) and quiet (0.7113) fills, since `d = delta +/- eta*q` moves only through inventory, which both models share - but the exponential model's informed-fill impact is nonetheless much larger in magnitude (-0.1285) than the queue model's AWARE case ends up protected against (-0.0731), since the queue model's distance widens far more (1.2764 vs 0.8311 quiet) in response to the SAME inventory, because the queue model's fill mechanics make informed episodes accumulate larger, more concentrated inventory swings (B+2's composition finding) that the skew formula reacts to more strongly.
- All reported `impact_h` and `markout_h` relationships hold at every horizon tested (5 to 50 ticks); the magnitude grows with horizon as expected (more time for the episode's drift to compound), and quiet-fill impact stays within bootstrap noise of zero throughout, as it should for a driftless baseline.

**Weaknesses:** the informed/quiet split uses the TRUE hidden `informed` flag (ACTIVE episodes only; lead-in ticks are excluded from "informed" here, unlike the detector's scoring convention elsewhere, which also counts lead-in) - purely for scoring, never fed to any strategy. Markouts are computed within-session only (a fill within h ticks of the session end is dropped for that horizon, not padded or wrapped across the boundary). The default-toxicity-only regime means this is one operating point, not a sweep; the over-compensation finding (aware's markout flips positive) is specific to this toxicity level and would be expected to reverse at higher toxicity, where C1's 2-D sweep already showed the P&L gap continuing to widen in aware's favour even as flow intensity rises - consistent with, but not directly re-derived from, this section's mechanism.

## UI pass (B+4 to B+8): tabs, book views, labels, n-labelling

- **Queue model is the dashboard default** (`build_dashboard(queue_fills=True)`; `--exponential` starts on the other model), so the live view shows the model the headline findings came from. `LiveMarketMaker`'s own default stays `queue_fills=False`, so no batch/equivalence contract moved. The checkbox queues a switch like any slider: it applies at the next step and opens a new epoch.
- **Queue parameters are live and epoch-tracked**: `QUEUE_KEYS = (depth_base, mo_rate, mo_size_mean, informed_mo_mult)` are part of `config()`, accepted by `request()`, and applied via `dataclasses.replace` on the frozen `QueueConfig`. `repro.live_script` / `batch_script` now emit the `QueueConfig` (and `want_queue=True` / the queue draws), and Freeze reproduces queue-model epochs. This is tested for sigma, depth_base and mo_rate changes.
- **`DisplayBook`** is a frozen read-only snapshot taken after each step. Under the queue model it is the fill-driving book itself: level prices, depth, our level, and how much rests ahead of us. Under the exponential model there is no book, so it shows the detector's synthetic book, labelled "SYNTHETIC BOOK · display only", with NaN queue fields. Extra `display_book()` calls are tested to change nothing.
- **Tabs (Market / Book / Strategies / Stats)** share one `update()`. The simulation steps, line data, statistics and control state update every frame whatever tab is showing; only heavy redraws (ladder, heatmap, bubbles) are skipped for hidden tabs. A test runs two dashboards with the same seed, flips tabs on one, and asserts identical P&L, inventory history and epoch tables.
- **Bubble chart**: colour is the detector score by default. The DEBUG toggle colours by the TRUE hidden state, is display-only, and is tested not to open an epoch or change `config()`.
- **n-labelling and the Stats tab's three tiers.** The panel is ordered by what matters: **tier 1** is the paired difference itself (large type, the research question), **tier 2** the supporting table (aware/naive mean ± se, Sharpe, max |q|, worst drawdown, n, aligned on the decimal point, plus the detector line with its denominators), **tier 3** the config as grouped key/value pairs in the smallest type, with any value differing from the project default marked `•`. Under-powering is carried by **one banner** (`UNDER-POWERED — n = x, need ~30+`) rather than per-number asterisks: an asterisk on every figure trained the eye to skip them, and one warning in the warning colour cannot be skipped. Below `MIN_N = 30` the banner shows and the tier-1 number takes the warning colour; at or above it the banner disappears and the number reads as normal text. Closed epochs are a compact table, each row greyed by its own n.
- **"detect nan%" fix**: `fmt()` renders any non-finite value as "—". A test asserts no "nan" reaches any text at 0 and 1 sessions with zero informed ticks.
- **Keys**: space or `p` pauses or resumes (`p` is removed from matplotlib's pan keymap).
- **Ordering note**: item 8 (the nan fix and visual pass) was done before B+3, because B+3 carried a stop condition and the visual pass tunes nothing.
- Widget tests drive real canvas events. Mutations reintroducing the nan bug, gating computation on the visible tab, or making the debug toggle a config change are each caught.

## B+3: queue-book calibration - Finding 1's second, independent route

Resolved: the stop condition fired, the user reviewed it, and the decision is to **keep the fill-rate-tuned defaults and report the incompatibility as the finding**. Recalibrating to realistic depth would re-run every result in a regime where almost nothing fills, and the book still would not be realistic (the cancellation ratio and the hump profile would remain wrong). **No default was changed.** The README now carries this as Finding 1's Route B rather than as a limitation.

**Two targets, kept separate on purpose:**
- (i) **Published order-book regularities. Citations CHECKED (2026-10-06); two were wrong and have been corrected - see "Citation check" below.** The *measured* quantities (sweep table, Pareto bound, cancellation ratio, profile shape) never depended on the citations; only the pass/fail labels against (i) did, and those labels survive the corrections.
  - Most market orders do not exhaust the depth at the best quote (**Hollifield, Miller & Sandas (2004)**, "Empirical Analysis of Limit Order Markets", *Review of Economic Studies* 71(4), 1027-1063, doi:10.1111/j.1467-937X.2004.00313.x: mean best-quote depth ≈ 9× mean market-order size, almost no orders exceeding available depth - **Stockholm Stock Exchange data, not US equities**). Operationalised as P(MO size > touch depth) < 0.5. **The threshold is this project's own, not a published number.**
  - Cancellation dominates execution (**Hasbrouck & Saar 2002**: 93% of INET limit orders cancelled). Operationalised as cancel/MO **volume** ≥ 10. **The threshold is this project's own, and carries a unit mismatch with its source - see below.**
  - Heavy-tailed order sizes with exponent ~1.5 (Gopikrishnan et al. 2000).
  - A hump-shaped mean depth profile (Bouchaud, Mézard & Potters 2002).
- (ii) **The exponential model's fill rate** at the default distance: 140·e^{-1.5·0.75}·0.005 = 0.227 per tick per side.

`scripts/calibrate_book.py` replays place → flow → trade → reanchor for the naive quoter at zero toxicity. A test asserts the replay matches `simulate_paths`' fills event for event.

**Result (n=400, seed 2024): the two targets cannot both be satisfied within this model.** Sweep of mean MO size / touch depth at MO rate 1:

| MO size / touch | P(sweep > best) | fill rate | satisfies |
|---|---|---|---|
| 0.5 | 37.6% | 0.0054 | (i) sweep only |
| 0.75 | 55.1% | 0.0421 | neither |
| 1.0 | 65.5% | 0.119 | neither |
| 1.44 | 75.7% | 0.287 | (ii) |
| ≥ 2 | ≥ 82% | > target | (ii) |

- The only (i)-satisfying point fills about 42× below target. Reaching 0.227 requires about 76% of market orders to sweep past the best quote.
- Heavy tails do not rescue it. With Pareto α=1.5 and median MO = touch, the analytic level-3 fill bound is 0.046 (back of queue) to 0.085 (front) for the provisional profile, and 0.063 to 0.096 for a flat profile. Both are below 0.227.
- The current provisional defaults sit at 90.8% sweep, cancel/MO 1.19, and fill rate 0.3465 (1.52× target).
- Cancel/MO volume is below 10 in every configuration tried.
- The mean depth profile rises monotonically (level 1 = 0.98× touch, level 15 = 10.5×). It shows only the rising side of the published hump and never decays.
- Wyart et al.'s spread ∝ volatility-per-trade relation is untestable here, because the spread is fixed by construction (exogenous mid).

**Disclosure:** two rows were first labelled "(i)-compliant" because they were designed at the NOMINAL touch depth. Measured, they sweep 78%, since the realised touch is thinner than nominal. They were relabelled before this note was written.

**Why this strengthens Finding 1 rather than repeating it.** Route A (the zero-fills result) came out of a *build accident*: a first guess at depth and market-order size that happened to produce no fills at all. Route B is *deliberate calibration* against published statistics, with the two targets defined in advance and held apart. The two routes share no parameters, no method and no motivation - one was found, the other was looked for - and they converge on the same conclusion: the exponential intensity at a 3-tick quoting distance implies a market where single market orders routinely sweep several levels. Two independent paths to one conclusion is a much stronger claim than either alone, which is why the README presents them as one finding with two routes rather than as two separate observations.

**Citation check (2026-10-06): what the recalled attributions got wrong.** The regularities were first written down from memory and flagged as pending verification. Verified, two of the four attributions were wrong:

1. **Farmer et al. (2004) was the wrong citation** for "most market orders do not exhaust the best quote" and is no longer used for it. That paper is about *large price changes being driven by gaps in the book rather than by large order volume*, and its 85% figure is **conditional on the order causing a price change** - a different quantity from the unconditional P(MO > touch depth) measured here. Using it would have been a quantity substitution, not just a loose attribution. It may still be cited for what it actually shows. The claim now rests on **Hollifield, Miller & Sandas (2004)** (full reference above), whose data is from the **Stockholm Stock Exchange** - a single European equity venue, which is why the venue-dependence paragraph below treats the regularity as a range rather than a constant. Note also that the paper has *three* authors; citing it as "Hollifield & Miller" drops Sandas.
2. **The cancellation claim is Hasbrouck & Saar (2002)** (93% of INET limit orders cancelled), **not the 2009 fleeting-orders paper**, which is about a different phenomenon.
3. **Gopikrishnan et al. (2000)** (size tail ~1.5) and **Bouchaud, Mézard & Potters (2002)** (hump profile) stand as cited.

**Unit mismatch, kept rather than papered over.** The 93% figure is a ratio by order **count** (≈13:1 cancels per execution); this project's threshold is a **volume** ratio ≥ 10. Those are different quantities and the published figure does not license the volume threshold. Restating the threshold in count terms is not possible here: this book has no individual orders at all - `flow()` cancels a *fraction of resting depth* per tick - so a count ratio is undefined in the model. The volume threshold is therefore kept and labelled explicitly as **this project's own operationalisation with no direct empirical backing**, and the measured 1.19× is read against it on that understanding.

**Target (i) is a tendency, not a law - and the evidence on both sides of it is single-venue.** The support is Hollifield, Miller & Sandas (2004) on the **Stockholm Stock Exchange**; the counterexample is a study of **eBay's** book, where nearly two-thirds (67%) of market orders completely empty the best offer, against the < 50% threshold used here. One European equity venue against one online auction platform. Neither establishes a general claim about markets, and between them the "share of market orders that clear the touch" runs from near zero to 67% - so the regularity is venue-dependent, and this project's threshold sits at the strict end of an observed *range*, not at a settled value. The finding is unaffected: 67% is still below the ~76% sweep rate needed to hit the exponential fill rate, and the heavy-tail bound caps the fill probability at 0.096 against 0.227 wherever the sweep threshold is drawn. **The incompatibility survives the strictest plausible reading of the empirical thresholds**, which is the right standard for it to be held to, since the thresholds are this project's own.

**Self-flags, kept visible in both documents:**
1. The two numeric thresholds (`P(MO > touch) < 0.5`, `cancel/MO >= 10`) are **this project's assumptions**, chosen to make qualitative claims testable - not literature values; the second also has the unit mismatch above.
2. Two of the four original attributions were wrong and were corrected after checking (above). The error was caught by verification, not by review of the write-up.
3. Two sweep rows were initially mislabelled "(i)-compliant" because they were designed against *nominal* touch depth; measured, the realised touch is thinner and they sweep 78%. Corrected before write-up, recorded rather than silently fixed.

## Known weaknesses so far
- Adverse selection is a stylised hidden-episode model (see adverse.py); Gaussian increments, no jumps; constant parameters; unit size; no fees, latency or queue position.
- Single-session results (as in the M1 figure) are anecdotes. Nothing about the research question can be concluded until M3+.
