# Remove legacy code, revamp price-return analysis (`dev/legacy_code_removal`)

This is the working plan for retiring the notebook-era code and extending `src/tools/price_return/`,
kept in the repo so work can continue from any machine or session. Update it as phases finish:
mark the phase ✅ with its commit hash in the table, and add a "Phase N notes" section with anything
a later phase needs to know.

## Picking this up

- **Setup:** Python 3.12 or newer. `pip install -e ".[ta,stats,dev]"` from the repo root, then
  `pytest`; all tests should pass offline.
- **Phases are the unit of work.** Do one phase when asked, then stop; do not start the next one
  unprompted. Commit and push at the end of each phase.
- **Stay within the plan.** Propose new deliverables as changes to this plan.
- **Statistics are checked against independent references** (hand formulas, closed forms, seeded
  simulations with known answers), never against the code under test.

## Context

`ta_tools` is finished and merged (`master` at `978ca40`, archived as `archive/02_ta_tools`). The
repo still carries the notebook-era code it grew out of: `src/tools/basic.py` and three `test_*`
notebooks built on it, plus the frozen `price_return_analysis_v0.1.ipynb`. The owner wants legacy
code removed wherever something equivalent already exists, and the price-return analysis
reviewed, fixed and extended with more detailed statistics.

The review found that almost all of the legacy code has a successor. The exception is the
**option helpers** in `basic.py` and the P&L grids in `test_es`, which exist nowhere else; per the
owner's choice these are **ported first, then deleted**. The review also found inconsistencies in
the existing `price_return` methods, which are fixed before new statistics are built on them.

**Owner's decisions (2026-09-28):** port the option helpers, then delete · keep v0.5 and add a
**new statistics notebook** · include all four statistics areas (distribution and tails,
dependence and volatility clustering, drawdowns and risk-adjusted returns, uncertainty on
probabilities) · remove v0.1 · **keep `config.py`** (it feeds the owner's local, gitignored
`dev/` notebooks).

## Legacy coverage map (what replaces what)

| Legacy | Successor | Action |
|---|---|---|
| `basic.px_plot` (Plotly layout helper) | `price_return/viz.py` styling | delete |
| `basic.consecutive_analysis` (prints, returns formatted strings) | `price_return.analysis.consecutive_analysis` (numeric, `n_obs`, `last_occurred`) | delete |
| `basic.sd_and_cond`, `profit_estimate`, `accepted_min_max`, `projected_min_max` | **none**, so port to `price_return/options.py` (Phase 1) | port, then delete |
| `test_es` cells 0–32, `test_ko` (the same notebook for KO) | `price_return_analysis.ipynb` (was v0.5) + `rare_case_run.ipynb` (ES=F is in `configs/tickers.yaml`; KO runs in v0.5 with `ticker='KO'`) | delete |
| `test_es` cells 33–36 (hand-tuned option P&L grids) | **none**, so it becomes the P&L example in the new notebook (Phase 5) | port, then delete |
| `test_ta_packages.ipynb` (SMA-20 in 3 libraries) | `ta_package_evaluation.ipynb` | delete |
| `price_return_analysis_v0.1.ipynb` (frozen monolith) | the `price_return` package + v0.5 + `tests/test_smoke.py` | delete |
| `matplotlib` dependency | used only by `basic.py` and `test_es`/`test_ko` | drop from `requirements.txt` and the `dev` extra |

## Methodology issues found in `price_return` (fixed in Phase 3)

1. **Three definitions of an n-day return.** `add_rolling_stats` (`PCT Change {d}`, annualised
   return) and `consecutive_analysis` (cumulative) *sum* simple daily returns;
   `analyze_cumulative` *compounds* them. They disagree most for large or long moves.
2. **Inconsistent frequency denominators.** Streak frequency divides by all days (`len(df)`, in
   `summarize_streaks` and `plot_streak_frequency`), cumulative frequency by complete windows
   (`n − w + 1`), and `consecutive_analysis` by the whole lookback, including the first
   `n_days − 1` windows that cannot be complete.
3. **Overlapping windows are counted as separate events.** One 6-day run is counted as five
   2-day streaks, so the "probabilities" overstate how often something *distinct* happens, and
   carry no uncertainty (a 0.4% probability from 500 days is 2 events).
4. **Mixed units in `Params`:** percent (`win_threshold`, `cum_thresholds`) next to decimals
   (`return_thresholds`). Renaming would break `configs/tickers.yaml`, so this is **documented,
   not changed**.

## Phases at a glance

| # | Phase | Deliverable |
|---|---|---|
| **0** ✅ | Branch and plan | `dev/legacy_code_removal` from `master` at `978ca40`; this plan as `docs/price_return_plan.md` |
| **1** ✅ | Port the option helpers | `src/tools/price_return/options.py` + tests — **done, `cd5677d`** |
| **2** ✅ | Remove legacy | delete 5 files, drop `matplotlib`, README + `ta_tools_plan.md` updated — **done, `cca0745`** |
| **3** ✅ | Fix existing methods | issues 1–3 above, each with a test that fails on the old code — **done, `0c847aa`** |
| **4** ✅ | Statistics module | `src/tools/price_return/stats.py` + tests; `scipy` as optional `stats` extra — **done, `eaab57c`** |
| **5** ✅ | Charts and new notebook | new `viz` functions; `notebooks/price_return_statistics.ipynb`; v0.5 refreshed — **done, `ed33638`** |
| **6** ✅ | Docs | README methodology, notebooks table, changelog; plan statuses — **done, `8cffb2e`** |

Working rules carried over from `docs/ta_tools_plan.md`: one phase per request, then stop;
statistical code is checked against **independent references** (hand formulas, closed forms,
seeded simulations with known answers), never against itself; each new test group is
mutation-checked; tests stay offline (`data_source='simulated'`); commit and push at the end of
each phase.

## Phase 1: port the option helpers (`price_return/options.py`)

Rebuilt as pure functions that **return tables instead of printing**, reusing
`add_rolling_stats` columns and `Params` (`trade_days`):

- `price_range(price, daily_vol, days=None, hours=None, session_hours=6.5, tick=None)`: one
  function for both `projected_min_max` (days ahead, `σ√d`) and `accepted_min_max` (hours left in
  the session, `σ√(h/6.5)`). Lognormal band `p·e^{±σ_t}`; `tick=0.1` reproduces the legacy
  ceil/floor rounding. Rejects `hours > session_hours`, as the legacy assert did.
- `move_probabilities(df, p, horizons=(1, 5, 10), fixed=None, scaled=None)`: replaces
  `sd_and_cond` plus the probability half of `profit_estimate`. For each horizon and each
  threshold method (**scaled**: a daily threshold × √n; **actual**: the latest rolling n-day std;
  **fixed**: a constant), it gives the share of the trailing `trade_days` windows moving above,
  below, or beyond ± the threshold. One tidy DataFrame instead of nested dicts.
- `expected_pnl(probabilities, pnl, contract_size)`: `(1 − p)·win + p·loss` per contract.
  `pnl` uses the legacy `{horizon: {direction: [win, loss]}}` shape, so the `test_es` grids paste
  in unchanged.
- **Tests:** closed-form values for `price_range`; `move_probabilities` against a hand-counted
  small series; `expected_pnl` arithmetic. A one-off **parity check against `basic.py`** on
  simulated data (same numbers, run before `basic.py` is deleted, recorded in the plan notes).

### Phase 1 notes

- `price_range`, `move_probabilities` and `expected_pnl` are exported from `price_return`. The
  legacy method names map as `th` → `scaled`, `act` → `actual`, `fix` → `fixed`; directions keep
  the legacy names (`above`, `below`, `exceeding`) so the `test_es` grids paste in unchanged.
- **Parity with `basic.py`, checked before its deletion:** the four `test_es` P&L grids
  (cells 33–36, verbatim) on 9 simulated series (seeds 1, 7, 42 × daily vol 0.6, 1.2, 2.5%), with
  `sd_th = 0.01`, `sd_fix = 0.02`, contract size 100. All 972 probabilities and expected P&Ls
  match `sd_and_cond` + `profit_estimate` exactly (max difference 0). All 108 `price_range`
  bands (4 prices × 3 vols × 5 day counts and 4 hour counts, `tick=0.1`) match
  `projected_min_max` / `accepted_min_max` exactly.
- One deliberate difference: `prob` divides by the **complete** windows in the trailing
  `trade_days` rows (`n_obs`), where `profit_estimate` always divided by `trade_days`. They agree
  whenever the history is longer than `trade_days` plus the horizon, as in every check above.
- `tick` rounding multiplies by `1 / tick` when that is a whole number, as `basic.py` did (`* 10`).
  Dividing by the tick instead moves a price already on the tick (4.1 → 4.0); a test covers this.
- Tests: 14 in `tests/test_price_return_stats.py` (closed-form bands, a hand-counted 7-row frame,
  the P&L arithmetic). Six seeded mutations were each caught. Suite: 170 (156 + 14).

## Phase 2: remove legacy

- Delete `src/tools/basic.py`, `notebooks/test_es.ipynb`, `test_ko.ipynb`,
  `test_ta_packages.ipynb`, `price_return_analysis_v0.1.ipynb`.
- Remove `matplotlib` from `requirements.txt` and the `dev` extra in `pyproject.toml`, after a
  `grep` confirms nothing else uses it.
- README: repo layout, the notebooks table, and the "`src/tools/basic.py` is superseded" section
  (replaced by a changelog entry pointing to `options.py`). The "two config mechanisms" section
  stays, since `config.py` stays.
- `docs/ta_tools_plan.md` → "Deferred, with reasons": mark the `basic.py` and old-notebook items
  resolved, pointing to this plan.
- **Check:** `grep` finds no reference to the removed files outside the changelogs, and `pytest`
  stays green.

### Phase 2 notes

- Deleted: `src/tools/basic.py`, `notebooks/test_es.ipynb`, `test_ko.ipynb`,
  `test_ta_packages.ipynb`, `price_return_analysis_v0.1.ipynb`. They remain in git history and on
  `archive/02_ta_tools`.
- `matplotlib` dropped from `requirements.txt` and the `dev` extra; a `grep` of `src/`, `tests/`
  and `notebooks/` finds no remaining import.
- README: layout (adds `options.py` and the two newer test files, drops the removed files), the
  notebooks table, the API table (adds the three `options` functions), the stale
  `src/tools/price_return.py` paths (now a package), and a 2026-09-29 changelog entry. The
  "`basic.py` is superseded" section is gone; the "two config mechanisms" section stays.
- `docs/ta_tools_plan.md`: the `basic.py` and old-notebook deferred items are marked done.
- Remaining mentions of the removed files are history: the changelogs, both plans, `options.py`'s
  provenance line, and one comment in `ta_package_evaluation.ipynb` (committed with outputs, left
  as is).
- Suite unchanged at 170; nothing imported the removed code.

## Phase 3: fix existing methods (numbers in v0.5 and `rare_case_run` will change)

- **Issue 1:** a single helper for the n-day return, **compounded** (what a position actually
  earns, and what `analyze_cumulative` already does), used by `add_rolling_stats`,
  `consecutive_analysis` and the annualised figures.
- **Issue 2:** every frequency divides by the number of **complete windows**.
- **Issue 3 (additive):** an `episodes` column alongside `count`, giving the number of distinct,
  non-overlapping occurrences, in `consecutive_analysis` and the streak and cumulative summaries.
- **Issue 4:** `Params` docstring and README state each field's unit.
- **Tests:** each fix gets a hand-built series where the old code gives the wrong answer and the
  new one the right answer; existing smoke tests updated where their expected numbers were built
  on the old definitions.

### Phase 3 notes

- **One n-day return:** `data.compound_returns(returns, n)` (exported), computed as
  `expm1(rolling_sum(log1p(r)))`; exact pass-through at `n = 1`. Used by `add_rolling_stats`
  (`PCT Change {d}` and `PCT Change Annualized`), `consecutive_analysis` (cumulative rows) and
  `analyze_cumulative`, which already compounded. Its results are unchanged: 48
  window/threshold cells on three simulated series match the old loop exactly, rounded values
  included. The annualized *volatility* is unchanged (daily std × √`trade_days`).
- **Complete-window denominators:** `summarize_streaks` and `plot_streak_frequency` divide by
  `len(df) − w + 1`; `consecutive_analysis` has a new `n_windows` column and `prob = count /
  n_windows`. `n_obs` keeps its meaning (trading days in the lookback). `options.
  move_probabilities` already worked this way.
- **Episodes:** `episodes` in `consecutive_analysis` and `analyze_cumulative`, `Win Episodes` /
  `Loss Episodes` in `summarize_streaks`, `≥x% Episodes` in `summarize_cumulative`. A run of
  consecutive qualifying window ends counts once. `detect_streaks` entries gain `end_pos`.
- **Units:** in the `Params` docstring and the README methodology, not renamed.
- **Tests:** six in `tests/test_price_return_stats.py`, each on a hand-built series. All six fail
  on the pre-Phase-3 code and pass now. Six seeded mutations were each caught. Suite: 176.
- **What shifts, measured on simulated series shaped like the configured asset classes** (seed
  11, drift 0.03%/day, thresholds as in `configs/tickers.yaml`). Real tickers could not be used,
  since Yahoo is blocked in this environment; the owner's next run of `rare_case_run` shows them.

  | series | annualized return | 3d win-streak freq | 3d drill rows | rare rows (all periods) | rare-event probs changed |
  |---|---|---|---|---|---|
  | FX-like, 0.5% vol, thr 0.1 | 17.83% → 19.17% | 8.67 → 8.68% | 27 → 27 | 176 → 173 | 346 of 832 |
  | index-like, 1.1%, thr 0.2 | 30.24% → 33.39% | 8.35 → 8.35% | 37 → 37 | 182 → 181 | 440 of 832 |
  | crude-like, 2.5%, thr 0.2 | 59.17% → 68.02% | 10.70 → 10.71% | 26 → 26 | 144 → 144 | 498 of 832 |
  | gas-like, 3.5%, thr 0.2 | 79.84% → 92.77% | 11.27 → 11.28% | 34 → 34 | 138 → 138 | 524 of 832 |

  Consecutive rows change only through the denominator (small rises, most at 20–30 days). The
  largest relative changes are extreme-tail cumulative rows resting on 1–3 windows. Compounding
  shrinks losses, so a −5%-in-20-days row can go from 2 windows to 0, and gains, so a +5%-in-2-days
  row from 2 to 3. Those rows were never reliable; Phase 4's bootstrap intervals quantify that.
- v0.5 and `rare_case_run` execute cleanly (v0.5 on simulated data; `rare_case_run` with a
  stubbed yfinance). Their markdown still describes frequencies as "% of all trading days" in
  places; that wording is refreshed in Phase 5.

## Phase 4: statistics module (`price_return/stats.py`)

Optional extra `stats = ["scipy"]` in `pyproject.toml` (also added to `requirements.txt` for
Colab). No `statsmodels`: the three tests it would supply are a few lines each and are easier to
verify written out.

- **Distribution and tails:** moments (mean, sd, skew, excess kurtosis); Jarque–Bera; Q-Q data
  against a normal and a fitted Student-t; Hill tail index for each tail; VaR and CVaR
  (historical, parametric normal, Cornish–Fisher) at 95/99% over 1/5/10-day compounded horizons.
- **Dependence and volatility clustering:** autocorrelation of returns and of squared returns;
  Ljung–Box Q; Lo–MacKinlay variance ratio with heteroskedasticity-robust z (trend vs mean
  reversion, the question v0.5's volatility chart already raises); Engle's ARCH-LM test.
- **Drawdowns and risk-adjusted returns:** drawdown series, maximum drawdown, its duration and
  recovery; Sharpe, Sortino and Calmar, plus their rolling `trade_days` versions.
- **Uncertainty on probabilities:** stationary block-bootstrap confidence intervals for the
  rare-event, streak and cumulative probabilities (block bootstrap because returns are
  dependent); model-implied probabilities from an i.i.d. normal and a fitted Student-t,
  compared with the empirical ones. *(As built, see the Phase 4 notes: dedicated intervals for
  the rare-event probabilities only; streak and cumulative frequencies can use the general
  `bootstrap_interval`, but have no function or table of their own.)*
- **Tests (independent references):** moments and JB on data with known values; Ljung–Box against
  the formula written out on a short series; variance ratio ≈ 1 and ARCH-LM not rejecting on
  i.i.d. data, but rejecting on a seeded GARCH(1,1); Hill recovering α on a Pareto sample; VaR on
  a normal sample against its quantile; drawdowns on a hand-built path; bootstrap intervals
  covering the true probability at roughly the nominal rate on seeded i.i.d. data.

### Phase 4 notes

- `price_return/stats.py`, 19 functions, all exported:
  - **distribution and tails:** `return_moments`, `jarque_bera`, `fit_student_t`, `qq_points`,
    `tail_index` (Hill), `value_at_risk` (historical, normal and Cornish–Fisher VaR and ES over
    1/5/10-day compounded horizons);
  - **dependence:** `autocorrelation` (returns, squared, absolute, with the 95% band),
    `ljung_box`, `variance_ratio` (Lo–MacKinlay with robust z*), `arch_lm`;
  - **drawdowns and ratios:** `drawdown_series`, `drawdown_table`, `max_drawdown`,
    `risk_ratios`, `rolling_risk`;
  - **uncertainty:** `stationary_bootstrap`, `bootstrap_interval`, `probability_intervals`
    (bootstrap bands on the rare-event probabilities), `model_probabilities` (what an i.i.d.
    normal and a fitted Student-t predict).
- **scipy is optional and lazy.** Only `fit_student_t`, `qq_points(dist='t')`, `ljung_box`,
  `arch_lm` and `model_probabilities` need it; they raise an install hint without it. Normal
  quantiles use the standard library's `NormalDist`, and the Jarque–Bera p-value is `exp(-JB/2)`
  (chi-squared with 2 df). Verified by importing with scipy blocked.
- **Bug caught while building:** `variance_ratio` first mixed two presentations of Lo–MacKinlay
  (the asymptotic δ(j), which carries a factor n, with the finite-sample φ, which has already
  divided by n), so the robust z* was about 50 times too small. On i.i.d. data it gave z = −0.45
  against z* = −0.01. Fixed by dropping n from δ(j); a test now requires z ≈ z* on
  constant-volatility data.
- `probability_intervals` computes the four event types for all thresholds at once on the
  bootstrap samples (cumulative sums, no rolling windows). Its point estimate equals
  `consecutive_analysis`'s `prob` to 1e-12 for 1, 3 and 10 days, which is tested. Default block
  mean is max(2 × n_days, 10), so most windows are resampled whole.
- `model_probabilities` fits both models to log returns: the normal is exact (log returns add
  over n days); the t is exact for consecutive events and simulated for cumulative ones.
- **Cornish–Fisher overshoots at 99%.** On a t(10) (excess kurtosis 1) it gives 0.0284 against a
  true 0.0276, still twice as close as the normal (0.0260). This is a property of the expansion,
  not a bug: the test checks it against the textbook formula with the population moments.
- **Tests:** 24 more in `tests/test_price_return_stats.py`. References: scipy's own moments and
  Jarque–Bera; Ljung–Box and the autocorrelation written out in the test; ARCH-LM at one lag equal
  to (n − 1) × corr²; the variance ratio against 1 + 2Σ(1 − k/q)ρ_k on AR(1) paths; Hill on a
  Pareto sample; VaR on hand-built and large normal samples; drawdowns and ratios on hand-built
  paths; bootstrap coverage of 88–99% over 200 datasets. 14 seeded mutations, each caught.
  Suite: 200.
- Two test designs were corrected, not loosened: the Hill check first used a *shifted* Pareto
  (Hill is scale- but not shift-invariant), and the Q-Q check first compared the single most
  extreme point, which is noise for any fit; it now compares the central 98%.

## Phase 5: charts and notebooks

- `viz.py` additions, in the existing Plotly style: Q-Q plot, ACF bars with confidence bands, an
  underwater (drawdown) chart, rolling Sharpe and volatility, empirical vs model probabilities
  with bootstrap error bars.
- **New `notebooks/price_return_statistics.ipynb`**, one ticker at a time via `Params`: data →
  distribution and tails → dependence → drawdowns → probabilities with uncertainty → **option
  P&L** (the `test_es` grids through `move_probabilities` + `expected_pnl`, plus `price_range`).
  Each section opens with the question it answers, in v0.5's style. It is committed without
  outputs for the owner to run on Yahoo data, and executed here on simulated data as a check.
- **v0.5:** fix its stale links (`src/tools/price_return.py` is now a package), show the new
  `episodes` column, and point to the statistics notebook.

### Phase 5 notes

- **Five charts in `viz.py`, all exported:** `plot_qq` (normal and Student-t against the
  45-degree line), `plot_autocorrelation` (returns and squared returns as two panels, with the
  95% band shaded), `plot_drawdown` (underwater, deepest troughs labelled), `plot_rolling_risk`
  (volatility and Sharpe as stacked panels), `plot_event_probabilities` (observed with bootstrap
  error bars against both models, log y axis, zero probabilities left out).
- **Styling:** the colour-blind-validated categorical palette already used by the `ta_tools`
  exploration notebook (blue, orange, aqua in fixed order; the validator passes all checks in
  light mode). Solid hairline gridlines, 2px lines, 8px markers, ink-coloured text, and direct
  labels on the model lines. Aqua is below 3:1 contrast on white, so its chart carries direct
  labels and sits next to its table. No chart overlays two y-scales; different units go in
  separate panels, which a test enforces.
- **Rendered and inspected** (Chromium screenshots of each chart on a GARCH series with t shocks).
  That caught three problems, all fixed before commit:
  1. the Q-Q markers' surface-coloured rings painted over the dense middle of the distribution;
  2. the autocorrelation band was never drawn, because plotly skips shapes on subplots that do not
     yet have traces (a regression test now requires the band in both panels);
  3. the log axis printed cluttered minor tick labels (now decades only, with a % suffix; the
     dashboard work later labels 1, 2 and 5 within each decade when the range is narrow).
- **`notebooks/price_return_statistics.ipynb`** (28 cells, committed without outputs): parameters
  from `configs/tickers.yaml` (default `ES=F`), then the four statistics areas and option sizing
  with the four `test_es` P&L grids verbatim. Each code cell opens with the question it answers.
  It avoids pandas `.style`, which needs jinja2 (not a dependency). It executes cleanly with a
  stubbed yfinance; the owner runs it on Yahoo data.
- **v0.5:** package links fixed (`src/tools/price_return/`), a pointer to the statistics
  notebook, and comments explaining complete-window frequencies, episodes and compounded
  cumulative moves. **`rare_case_run`:** "% of all trading days" becomes "% of complete windows".
  Both were edited in their own JSON layout (12 changed lines), and both execute cleanly.
- **Tests:** 3 chart tests (trace names, log axis, trough labels, no overlaid y-scales, the band in
  both panels, zero probabilities left out); 3 seeded mutations caught. Suite: 203.

## Phase 6: docs

README: methodology sections for the new statistics and the fixed definitions, layout, the
notebooks table (now without v0.1 and the `test_*` notebooks), setup (`.[stats]`), changelog.
Plan statuses and commit hashes, as in the `ta_tools` plan.

### Phase 6 notes

- README: setup gains the `stats` extra, and says which five functions need scipy; the
  Methodology gains a "Statistics" section (the four groups, with the caveats: Cornish–Fisher's
  overshoot at 99%, overlapping multi-day windows, the bootstrap's block length, what the i.i.d.
  models can and cannot say); layout adds `stats.py` and the statistics notebook; the notebooks
  table adds it; the API table adds the 19 statistics functions and 5 charts, plus 4 older
  multi-ticker functions it had never listed. A script now confirms the table and `__all__`
  match exactly. The Tests section, which still said the methods were "not yet verified against
  independent reference values", describes what the suites now check.
- **Final verification:** a fresh Python 3.12 environment with `pip install -e ".[ta,stats,dev]"`
  passes all 203 tests, without matplotlib installed.
- **All phases done.** Merged into `master` through PR #9 (`823529f`). The owner's first
  run of `price_return_statistics.ipynb` and `rare_case_run` on Yahoo data will show the real-ticker
  numbers that could not be produced here (Yahoo is blocked in this environment).

### After the plan: one price-return analysis notebook

- At the owner's request, `price_return_analysis_v0.5.ipynb` is renamed
  `price_return_analysis.ipynb`, so there is a single version (v0.1 was removed in Phase 2). "v0.5"
  in the phase notes above refers to this notebook. References updated in the README (layout,
  notebooks table, methodology, this work's changelog entry), the statistics notebook's links and
  text, and the package docstring. The rename is a `git mv`, so the file's history follows it.

## Critical files

- New: `src/tools/price_return/options.py`, `src/tools/price_return/stats.py`,
  `notebooks/price_return_statistics.ipynb`, `docs/price_return_plan.md`,
  `tests/test_price_return_stats.py` (Phases 1, 3 and 4 tests; renamed `test_price_return.py`
  on 2026-09-30).
- Modified: `src/tools/price_return/{analysis,data,viz,params,__init__}.py`,
  `tests/test_smoke.py`, `notebooks/price_return_analysis.ipynb` (renamed from v0.5), `pyproject.toml`,
  `requirements.txt`, `README.md`, `docs/ta_tools_plan.md`.
- Deleted: `src/tools/basic.py`, `notebooks/test_{es,ko,ta_packages}.ipynb`,
  `notebooks/price_return_analysis_v0.1.ipynb`.

## Verification

- `pip install -e ".[ta,stats,dev]"` on Python 3.12, then `pytest`: all green offline, with
  counts recorded per phase.
- Seeded mutations for each new test group (as in `ta_tools` Phases 7–8).
- The Phase 1 parity check against `basic.py` runs before the deletion in Phase 2.
- After Phase 2: `grep -rn "basic\|test_es\|test_ko\|test_ta_packages\|v0.1"` is clean outside
  the changelogs.
- Notebooks: executed here on simulated data; the owner runs the statistics notebook, v0.5 and
  `rare_case_run` on Yahoo data. Numbers that change in Phase 3 are listed in the plan notes, so
  the change is explained rather than surprising.
- A pull request into `master` when all phases are done, or earlier if the owner wants.
