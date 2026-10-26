from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from collections.abc import Callable
from typing import NamedTuple, cast

from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA

from data import loadTreasury, loadSpread
from model.var import ESGVar
from util import Panels, Style, plot

import pandas as pd
import warnings

MAX_LAGS = 24


class Element(NamedTuple):
    data: pd.DataFrame | Panels
    fileName: str
    title: str


treasury = loadTreasury()
spread = loadSpread()
tPCA = PCA(3)
tScore = pd.DataFrame(
    tPCA.fit_transform(treasury),
    index=treasury.index,
    columns=["TPC1", "TPC2", "TPC3"],
)
tLoading = pd.DataFrame(
    tPCA.components_.T,
    index=treasury.columns,
    columns=["TPC1", "TPC2", "TPC3"],
)
sPCA = PCA(3)
sScore = pd.DataFrame(
    sPCA.fit_transform(spread),
    index=spread.index,
    columns=["SPC1", "SPC2", "SPC3"],
)
sLoading = pd.DataFrame(
    sPCA.components_.T,
    index=spread.columns,
    columns=["SPC1", "SPC2", "SPC3"],
)
cScore = pd.concat([tScore, sScore], axis=1).asfreq("MS").dropna()
cLoading = pd.concat([tLoading, sLoading], axis=1)

cScoreDiff = cScore.diff().dropna()
wScoreDiff: list[warnings.WarningMessage] = []
with warnings.catch_warnings(record=True) as wScoreDiff:
    warnings.simplefilter("always")
    fit = cast(Callable[..., VARResultsWrapper], VAR(cScoreDiff).fit)
    model = ESGVar(fit(maxlags=MAX_LAGS, ic="aic", trend="c"))
print("\nESG using differenced scores:")
for w in wScoreDiff:
    print(f"\t{w.message}")

forecast = model.generateForecast()
forecastLevels = forecast.groupby(level="scenario").cumsum() + cScore.iloc[-1]
tForecast = pd.DataFrame(
    tPCA.inverse_transform(forecastLevels[tScore.columns]),
    index=forecastLevels.index,
    columns=treasury.columns,
)
sForecast = pd.DataFrame(
    sPCA.inverse_transform(forecastLevels[sScore.columns]),
    index=forecastLevels.index,
    columns=spread.columns,
)
cForecast = tForecast + sForecast

elements: tuple[Element, ...] = (
    Element(
        pd.concat(
            {"Treasury": treasury[0.5], "Corporate": treasury[0.5] + spread[0.5]},
            axis=1,
        ),
        "six_month",
        title="6-Month Interest Rates",
    ),
    Element(
        pd.concat(
            {"Treasury": treasury[2.0], "Corporate": treasury[2.0] + spread[2.0]},
            axis=1,
        ),
        "two_year",
        title="2-Year Interest Rates",
    ),
    Element(
        pd.concat(
            {"Treasury": treasury[10.0], "Corporate": treasury[10.0] + spread[10.0]},
            axis=1,
        ),
        "ten_year",
        title="10-Year Interest Rates",
    ),
    Element(
        cScore,
        "cScore",
        title="Zero-Coupon Treasury Yield and Credit Spread PCA Scores",
    ),
    Element(
        cLoading,
        "cLoading",
        title="Zero-Coupon Treasury Yield and Credit Spread PCA Loadings",
    ),
    Element(
        cScoreDiff,
        "cScoreDiff",
        title="Zero-Coupon Treasury Yield and Credit Spread PCA Score Differences",
    ),
    Element(
        (
            (tForecast[10.0].unstack("scenario"), "Treasury", Style.PATHS),
            (cForecast[10.0].unstack("scenario"), "Corporate", Style.PATHS),
        ),
        "10_year_interest_rate_paths",
        title="10-Year Spot Rate Forecast",
    ),
)
for e in elements:
    plot(*e)
