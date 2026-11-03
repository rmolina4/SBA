from data import loadTreasury, loadCorporate
from util import *
from plot import *

import pandas as pd

BACKTEST_MONTHS = 36
MATURITIES = (
    (0.5, "six_month", "6-Month"),
    (2.0, "two_year", "2-Year"),
    (10.0, "ten_year", "10-Year"),
)
MATURITY = MATURITIES[2][0]

# LOAD DATA
treasury = loadTreasury()
corporate = loadCorporate()
spread = (corporate - treasury).dropna()

# BACKTEST
cScore, cLoading, tPCA, sPCA = runIPCA(
    treasury.iloc[:-BACKTEST_MONTHS], spread.iloc[:-BACKTEST_MONTHS]
)
esg = createESG(cScore.diff().dropna())
tBacktestMean, cBacktestMean, backtestHistoryDates = forecastIMean(
    esg, cScore, tPCA, sPCA, treasury.columns, BACKTEST_MONTHS
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

# FORECAST
(
    cScore,
    cScoreDiff,
    cLoading,
    tForecast,
    cForecast,
    tMean,
    cMean,
    historyDates,
) = forecastInterest(treasury, spread)

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
):
    plot(
        df,
        filename,
        f"Treasury Yield and Credit Spread PCA {title}",
    )

# FORECAST PLOT DATA
tHistory = treasury.loc[historyDates]
cHistory = corporate.loc[historyDates]
tMean10 = pd.concat([tHistory[MATURITY].iloc[-1:], tMean[MATURITY]]).rename("VAR")
cMean10 = pd.concat([cHistory[MATURITY].iloc[-1:], cMean[MATURITY]]).rename("VAR")

tPath10 = tForecast[MATURITY].unstack("scenario")
cPath10 = cForecast[MATURITY].unstack("scenario")
tPath10 = tPath10.reindex(historyDates.append(tPath10.index))
cPath10 = cPath10.reindex(historyDates.append(cPath10.index))
tPath10.loc[historyDates, :] = tHistory[MATURITY]
cPath10.loc[historyDates, :] = cHistory[MATURITY]

# BACKTEST PLOT DATA
tObserved10 = treasury.loc[backtestHistoryDates[0] :, MATURITY].rename("Observed")
cObserved10 = corporate.loc[backtestHistoryDates[0] :, MATURITY].rename("Observed")
tBacktestMean10 = pd.concat(
    [treasury.loc[backtestHistoryDates[-1:], MATURITY], tBacktestMean[MATURITY]]
).rename("VAR")
cBacktestMean10 = pd.concat(
    [corporate.loc[backtestHistoryDates[-1:], MATURITY], cBacktestMean[MATURITY]]
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
