from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA

from data import loadTreasury, loadSpread
from model.var import ESGVar

import pandas as pd

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

esgCScore = ESGVar(VAR(cScore).fit(maxlags=24, ic="aic", trend="c"))
esgCScoreDiff = ESGVar(VAR(cScoreDiff).fit(maxlags=24, ic="aic", trend="c"))  # type: ignore
