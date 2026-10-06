from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA

from data import loadTreasury, loadSpread
from model.var import ESGVar
from util import plot

import pandas as pd
import warnings

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
cScoreDiff = cScore.diff().dropna()

esgWarnings1: list[warnings.WarningMessage] = []
esgWarnings2: list[warnings.WarningMessage] = []

with warnings.catch_warnings(record=True) as esgWarnings1:
    warnings.simplefilter("always")
    esg1 = ESGVar(VAR(cScore).fit(maxlags=MAX_LAGS, ic="aic", trend="c"))

with warnings.catch_warnings(record=True) as esgWarnings2:
    warnings.simplefilter("always")
    esg2 = ESGVar(VAR(cScoreDiff).fit(maxlags=MAX_LAGS, ic="aic", trend="c"))

print("ESG using scores:")
for warning in esgWarnings1:
    print(f"\t{warning.message}")

print("\nESG using differenced scores:")
for warning in esgWarnings2:
    print(f"\t{warning.message}")

elements: tuple[tuple[pd.DataFrame, str, str], ...] = (
    (cScore, "cScore", "cScore"),
)

for e in elements:
    plot(*e)
