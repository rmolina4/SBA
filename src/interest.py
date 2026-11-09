from typing import Final

from data import loadTreasury, loadCorporate
from util import *
from plot import *

import pandas as pd

EVAL_DIR: Final = OUT_DIR / "evaluation"
MATURITIES: Final = (
    (0.5, "six_month", "6-Month"),
    (2.0, "two_year", "2-Year"),
    (10.0, "ten_year", "10-Year"),
)
MATURITY: Final = MATURITIES[2][0]
BACKTEST_HORIZON: Final = 24
BACKTEST_MONTHS: Final = 60
FORECAST_HORIZON: Final = 60

# LOAD DATA
treasury = loadTreasury()
corporate = loadCorporate()
spread = (corporate - treasury).dropna()

# BACKTEST
backtestData: list[pd.DataFrame] = []
backtestKeys: list[tuple[int, str]] = []
backtestPlots: dict[
    tuple[int, str],
    tuple[pd.DataFrame, pd.Index],
] = {}
for cutoff in range(BACKTEST_HORIZON, BACKTEST_MONTHS + 1):
    (
        *_,
        tPath,
        cPath,
        tMean,
        cMean,
        historyDates,
        forecastDates,
    ) = forecastInterest(
        treasury.iloc[:-cutoff], spread.iloc[:-cutoff], BACKTEST_HORIZON
    )
    for curve, observed, mean, path in (
        ("T", treasury, tMean, tPath),
        ("C", corporate, cMean, cPath),
    ):
        lastObserved = observed.loc[historyDates[-1:]].iloc[0]
        observed = observed.loc[forecastDates]
        varSE = (observed - mean).pow(2)
        bmSE = (observed - lastObserved).pow(2)

        lower = path.groupby(level="date").quantile(0.025)
        upper = path.groupby(level="date").quantile(0.975)
        coverage = (observed >= lower) & (observed <= upper)

        backtestData.append(
            pd.concat(
                {
                    "VAR_SE": varSE,
                    "BM_SE": bmSE,
                    "Coverage": coverage,
                },
                names=["metric"],
            )
        )
        backtestKeys.append((cutoff, curve))
        backtestPlots[cutoff, curve] = (mean, historyDates)

# EVALUATION
backtestResult = pd.concat(
    backtestData,
    keys=backtestKeys,
    names=["cutoff", "curve"],
)
backtestMetrics = (
    backtestResult.groupby(level=["cutoff", "curve", "metric"])
    .mean()
    .rename(index={"VAR_SE": "VAR_MSE", "BM_SE": "BM_MSE"}, level="metric")
)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
backtestResult.to_csv(EVAL_DIR / "backtestResult.csv")
backtestMetrics.to_csv(EVAL_DIR / "backtestMetrics.csv")

# FORECAST
(
    cScore,
    cScoreDiff,
    cLoading,
    cResidual,
    tPath,
    cPath,
    tMean,
    cMean,
    historyDates,
    forecastDates,
) = forecastInterest(treasury, spread, FORECAST_HORIZON)

# FORECAST PLOT DATA
tHistory = treasury.loc[historyDates, MATURITY]
cHistory = corporate.loc[historyDates, MATURITY]
tMean10 = pd.concat([tHistory.iloc[-1:], tMean[MATURITY]]).rename("VAR")
cMean10 = pd.concat([cHistory.iloc[-1:], cMean[MATURITY]]).rename("VAR")
tPath10 = tPath[MATURITY].unstack("scenario")
cPath10 = cPath[MATURITY].unstack("scenario")
tPath10 = tPath10.reindex(historyDates.append(forecastDates))
cPath10 = cPath10.reindex(historyDates.append(forecastDates))
tPath10.loc[historyDates, :] = tHistory
cPath10.loc[historyDates, :] = cHistory

# BACKTEST PLOT DATA
tBacktestMean, backtestHistoryDates = backtestPlots[BACKTEST_MONTHS, "T"]
cBacktestMean, _ = backtestPlots[BACKTEST_MONTHS, "C"]
tObserved10 = treasury.loc[backtestHistoryDates[0] :, MATURITY].rename("Observed")
cObserved10 = corporate.loc[backtestHistoryDates[0] :, MATURITY].rename("Observed")
tBacktestMean10 = pd.concat(
    [tObserved10.loc[backtestHistoryDates[-1:]], tBacktestMean[MATURITY]]
).rename("VAR")
cBacktestMean10 = pd.concat(
    [cObserved10.loc[backtestHistoryDates[-1:]], cBacktestMean[MATURITY]]
).rename("VAR")

for tLayers, cLayers, filename, title in (
    (
        (
            Layer(tPath10, Style.PATHS, "C0"),
            Layer(tMean10, Style.DASHED, "#13476c"),
        ),
        (
            Layer(cPath10, Style.PATHS, "C1"),
            Layer(cMean10, Style.DASHED, "#994c08"),
        ),
        "paths",
        "Forecast",
    ),
    (
        (
            Layer(tObserved10, Style.LINE, "C0"),
            Layer(tBacktestMean10, Style.DASHED, "C2"),
        ),
        (
            Layer(cObserved10, Style.LINE, "C1"),
            Layer(cBacktestMean10, Style.DASHED, "C2"),
        ),
        "backtest",
        "Backtest",
    ),
):
    plot(
        (Panel("Treasury", tLayers), Panel("Corporate", cLayers)),
        f"10_year_spot_rate_{filename}",
        f"10-Year Spot Rate {title}",
    )

for maturity, filename, title in MATURITIES:
    plot(
        pd.DataFrame(
            {"Treasury": treasury[maturity], "Corporate": corporate[maturity]}
        ),
        filename,
        f"{title} Spot Rates",
    )

for df, filename, title in (
    (cScore, "cScore", "Scores"),
    (cLoading, "cLoading", "Loadings"),
    (cScoreDiff, "cScoreDiff", "Score Differences"),
    (cResidual, "cResidual", "Residuals"),
):
    plot(
        df,
        filename,
        f"Treasury Yield and Credit Spread PCA {title}",
    )
