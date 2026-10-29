from data import loadTreasury, loadCorporate
from util import *
from plot import *

import pandas as pd

BACKTEST_MONTHS = 36


class Element(NamedTuple):
    data: pd.DataFrame | Panels
    fileName: str
    title: str


treasury = loadTreasury()
corporate = loadCorporate()
spread = (corporate - treasury).dropna()

cScore, cLoading, tPCA, sPCA = runIPCA(
    treasury.iloc[:-BACKTEST_MONTHS], spread.iloc[:-BACKTEST_MONTHS]
)
esg = createESG(cScore.diff().dropna())
maturities = treasury.columns
tBacktestMean, cBacktestMean = forecastIMean(
    esg, cScore, tPCA, sPCA, maturities, BACKTEST_MONTHS
)
tMSE = (
    (treasury.iloc[-BACKTEST_MONTHS:] - tBacktestMean)
    .pow(2)
    .mean()
    .rename("Treasury MSE")
)
cMSE = ((corporate.iloc[-BACKTEST_MONTHS:] - cBacktestMean).pow(2).mean()).rename(
    "Corporate MSE"
)
print(tMSE)
print(cMSE)

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
            {"Treasury": treasury[0.5], "Corporate": corporate[0.5]},
            axis=1,
        ),
        "six_month",
        "6-Month Interest Rates",
    ),
    Element(
        pd.concat(
            {"Treasury": treasury[2.0], "Corporate": corporate[2.0]},
            axis=1,
        ),
        "two_year",
        "2-Year Interest Rates",
    ),
    Element(
        pd.concat(
            {"Treasury": treasury[10.0], "Corporate": corporate[10.0]},
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
        "10_year_spot_rate_paths",
        "10-Year Spot Rate Forecast",
    ),
    Element(
        (
            Panel(
                treasury.iloc[-BACKTEST_MONTHS:][10.0].rename("Observed"),
                "Treasury",
                Style.NORMAL,
                mean=tBacktestMean[10.0],
            ),
            Panel(
                (corporate.iloc[-BACKTEST_MONTHS:][10.0]).rename("Observed"),
                "Corporate",
                Style.NORMAL,
                mean=cBacktestMean[10.0],
            ),
        ),
        "10_year_spot_rate_backtest",
        f"10-Year Spot Rate Backtest",
    ),
)
for e in elements:
    plot(*e)
