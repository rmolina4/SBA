from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA

from data import loadTreasury, loadSpread
from model.var import ESGVar
from util import plot

import pandas as pd
import warnings

Element = tuple[pd.DataFrame, str, str]
MAX_LAGS = 24

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

wScore: list[warnings.WarningMessage] = []
with warnings.catch_warnings(record=True) as wScore:
    warnings.simplefilter("always")
    ESGVar(VAR(cScore).fit(maxlags=MAX_LAGS, ic="aic", trend="c"))
print("ESG using scores:")
for w in wScore:
    print(f"\t{w.message}")

cScoreDiff = cScore.diff().dropna()
wScoreDiff: list[warnings.WarningMessage] = []
with warnings.catch_warnings(record=True) as wScoreDiff:
    warnings.simplefilter("always")
    ESGVar(VAR(cScoreDiff).fit(maxlags=MAX_LAGS, ic="aic", trend="c"))
print("\nESG using differenced scores:")
for w in wScoreDiff:
    print(f"\t{w.message}")

elements: tuple[Element, ...] = (
    (
        pd.concat(
            {"Treasury": treasury[0.5], "Corporate": treasury[0.5] + spread[0.5]},
            axis=1,
        ),
        "6-Month Interest Rates",
        "6month",
    ),
    (
        pd.concat(
            {"Treasury": treasury[2.0], "Corporate": treasury[2.0] + spread[2.0]},
            axis=1,
        ),
        "2-Year Interest Rates",
        "2year",
    ),
    (
        pd.concat(
            {"Treasury": treasury[10.0], "Corporate": treasury[10.0] + spread[10.0]},
            axis=1,
        ),
        "10-Year Interest Rates",
        "10year",
    ),
    (cScore, "Zero-Coupon Treasury Yield and Credit Spread PCA Scores", "cScore"),
    (
        cLoading,
        "Zero-Coupon Treasury Yield and Credit Spread PCA Loadings",
        "cLoading",
    ),
    (
        cScoreDiff,
        "Zero-Coupon Treasury Yield and Credit Spread PCA Score Differences",
        "cScoreDiff",
    ),
)
for e in elements:
    plot(*e)
