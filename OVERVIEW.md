# SBA — Project Overview

## What it is

An **asset-liability project whose deliverable is an investment strategy** —
plus a written report. The aim is to build scenarios that are both
statistically plausible and economically coherent, then turn what those
scenarios say about returns, risk, correlations and macro conditions into
portfolio allocation decisions.

```
market + mortality data  ->  scenarios
                         ->  probabilities, expected returns, risk, correlations
                         ->  allocation
```

Two blocks of risk feed the scenario engine:

- **Rates and credit.** A pension-style liability is the present value of
  future payments, so the discount curve drives its value. The Treasury curve
  and the HQM corporate curve (Treasury plus a credit spread) are both
  modeled. These are also the raw material for bond returns on the asset side.
- **Longevity.** How long retirees live determines the cash flows being
  discounted. SOA's Pri-2012 tables and the MP-2021 improvement scale are
  projected forward into survival curves.

The mortality side is what makes this asset-*liability* rather than asset-only:
the portfolio is being chosen against a liability, not in isolation.

## The data

Everything is vendored in `data/` — about 1.7 MB of spreadsheets from the U.S.
Treasury and the Society of Actuaries:

| What | Coverage |
|---|---|
| Treasury nominal spot curve | monthly, 2003 – present, maturities 0.5–100y in half-year steps |
| HQM corporate bond spot curve | monthly, 1999 – present, same maturity grid |
| Pri-2012 mortality (amount-weighted) | retiree death rates by age and sex |
| MP-2021 improvement scale | how fast mortality is improving, by age and calendar year |

Both curves sit on an identical 200-point maturity grid, so the credit spread
is a clean subtraction. The curve files are five-year blocks, and dropping a
new one into `data/` automatically extends the series.

## What the code does today

Roughly 320 lines in `src/`. `data.py` loads, cleans and exports; `main.py`
runs the analysis top to bottom.

**1. Build survival curves.** `loadSurvival()` takes the 2012 base mortality
rates and projects them forward year by year using the improvement scale, then
accumulates them into survival probabilities by age and sex.

**2. Compress the curves.** A yield curve has 200 maturities that move almost
in lockstep. Principal component analysis reduces each curve to three factors
capturing well over 99% of the movement, with the standard interpretations:

- **Level** — the whole curve shifts up or down
- **Slope** — short and long ends move apart
- **Curvature** — the middle bows

Done twice: once on the Treasury curve (`TPC1–3`), once on the credit spread
(`SPC1–3`). Six factors now stand in for 400 numbers per month.

**3. Check stationarity.** Augmented Dickey-Fuller tests confirm the factors
wander in levels but are well-behaved in first differences. So the model is
built on monthly *changes*.

**4. Fit the dynamics.** A vector autoregression on the six differenced
factors, lag length chosen by AIC. This captures how the factors lead and feed
back on each other — a steepening today predicting a level move next quarter,
or spreads widening when Treasuries rally. A portmanteau test checks whether
anything systematic is left in the residuals.

**5. Export artifacts.** Figures as PNG and LaTeX matrix fragments to `out/`,
for the write-up.

## What's missing

The gap between the code and the objective:

- **Simulation.** `SCENARIO_COUNT = 10000` and `HORIZON_MONTHS = 60` are
  declared and never used. Nothing is sampled anywhere. The VAR coefficients
  and residual covariance are the inputs a simulator would need, and they are
  sitting right there.
- **Asset returns.** No return series for any asset class. Worth knowing:
  Treasury and corporate bond returns are *fully derivable* from the spot
  curves already in hand — reprice a bond a month later, one month shorter, off
  the next month's curve. Equities and cash would need new data.
- **Macro data.** No inflation, growth or policy-rate series anywhere in
  `data/`, so the "macroeconomic conditions" part of the brief can't currently
  be computed from this repo.
- **Liability cash flows.** Survival probabilities exist, but there's no
  population, benefit amount or payment stream — so no liability number yet.
- **The allocation itself.** No optimizer, no backtest.

## Two modeling notes worth knowing

**The survival curves are period, not cohort.** They project mortality to a
given calendar year and then read down the age axis within that year, rather
than following a single birth cohort diagonally through time. They also start
at the first retiree age, so they express survival *conditional on reaching
that age*, not from birth. Both are reasonable; they just need to match
whatever the liability model assumes.

**Scenario coherence will need explicit attention.** A model fitted to changes
behaves like a random walk in levels, so over five years an unconstrained
simulation will eventually produce negative rates and — because Treasury and
spread are modeled separately — corporate curves below Treasury, which is
economically impossible. Constraints aren't a nicety here; they're part of
getting the scenario set right.

## Practical notes

- Run with `python src/main.py`. Output lands in `out/` (gitignored).
- Environment is pixi, but `pixi.toml` and `pixi.lock` are **gitignored**, so
  the repo doesn't pin its own environment. It currently declares
  `osx-arm64` only, so `pixi install` will fail on Linux or Windows until the
  platform is added.
- Recent pins: Python 3.14, pandas 3.0, scikit-learn 1.9, statsmodels 0.15.
- No tests, no CI. It's a research script with a report attached, and reads
  like one.
- Naming is camelCase throughout (`loadSurvival`, `cScoreDiff`) — unusual for
  Python, but applied consistently.
