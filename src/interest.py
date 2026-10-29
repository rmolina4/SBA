from data import loadTreasury, loadCorporate
from util import *
from plot import *

import pandas as pd

BACKTEST_MONTHS = 36


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

for maturity, filename, title in (
    (0.5, "six_month", "6-Month"),
    (2.0, "two_year", "2-Year"),
    (10.0, "ten_year", "10-Year"),
):
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
):
    plot(
        df,
        filename,
        f"Treasury Yield and Credit Spread PCA {title}",
    )

plot(
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
)

plot(
    (
        Panel(
            treasury.iloc[-BACKTEST_MONTHS:][10.0].rename("Observed"),
            "Treasury",
            Style.NORMAL,
            mean=tBacktestMean[10.0],
        ),
        Panel(
            corporate.iloc[-BACKTEST_MONTHS:][10.0].rename("Observed"),
            "Corporate",
            Style.NORMAL,
            mean=cBacktestMean[10.0],
        ),
    ),
    "10_year_spot_rate_backtest",
    "10-Year Spot Rate Backtest",
)
