from data import loadTreasury, loadSpread
from util import *
from plot import *

import pandas as pd

BACKTEST_MONTHS = 36


class Element(NamedTuple):
    data: pd.DataFrame | Panels
    fileName: str
    title: str


treasury = loadTreasury()
spread = loadSpread()

cScore, cLoading, tPCA, sPCA = runPCA(
    treasury.iloc[:-BACKTEST_MONTHS], spread.iloc[:-BACKTEST_MONTHS]
)
esg = createESG(cScore.diff().dropna())
maturities = treasury.columns
tMean, cMean = forecastIMean(esg, cScore, tPCA, sPCA, maturities, BACKTEST_MONTHS)
tRMSE = ((treasury.iloc[-BACKTEST_MONTHS:] - tMean).pow(2).mean() ** 0.5).rename(
    "Treasury RMSE (pp)"
)
cRMSE = (
    (
        (treasury.iloc[-BACKTEST_MONTHS:] + spread.iloc[-BACKTEST_MONTHS:] - cMean)
        .pow(2)
        .mean()
    )
    ** 0.5
).rename("Corporate RMSE (pp)")
print(tRMSE)
print(cRMSE)

(
    cScore,
    cScoreDiff,
    cLoading,
    tForecast,
    cForecast,
    tMean,
    cMean,
    tHistory,
    cHistory,
) = forecastInterest(treasury, spread)

elements: tuple[Element, ...] = (
    Element(
        pd.concat(
            {"Treasury": treasury[0.5], "Corporate": treasury[0.5] + spread[0.5]},
            axis=1,
        ),
        "six_month",
        "6-Month Interest Rates",
    ),
    Element(
        pd.concat(
            {"Treasury": treasury[2.0], "Corporate": treasury[2.0] + spread[2.0]},
            axis=1,
        ),
        "two_year",
        "2-Year Interest Rates",
    ),
    Element(
        pd.concat(
            {"Treasury": treasury[10.0], "Corporate": treasury[10.0] + spread[10.0]},
            axis=1,
        ),
        "ten_year",
        "10-Year Interest Rates",
    ),
    Element(
        cScore,
        "cScore",
        "Zero-Coupon Treasury Yield and Credit Spread PCA Scores",
    ),
    Element(
        cLoading,
        "cLoading",
        "Zero-Coupon Treasury Yield and Credit Spread PCA Loadings",
    ),
    Element(
        cScoreDiff,
        "cScoreDiff",
        "Zero-Coupon Treasury Yield and Credit Spread PCA Score Differences",
    ),
    Element(
        (
            Panel(
                tForecast[10.0].unstack("scenario"),
                "Treasury",
                Style.PATHS,
                tHistory[10.0],
                tMean[10.0],
            ),
            Panel(
                cForecast[10.0].unstack("scenario"),
                "Corporate",
                Style.PATHS,
                cHistory[10.0],
                cMean[10.0],
            ),
        ),
        "10_year_interest_rate_paths",
        "10-Year Spot Rate Forecast",
    ),
)
for e in elements:
    plot(*e)
