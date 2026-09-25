import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from data import *

SCENARIO_COUNT = 10000
HORIZON_MONTHS = 60
SEED = 42

mortality = loadMortality()
improvement = loadImprovement()
treasury = loadTreasury()
corporate = loadCorporate()
spread = corporate.loc[treasury.index, :] - treasury

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
    index=treasury.index,
    columns=["SPC1", "SPC2", "SPC3"],
)
sLoading = pd.DataFrame(
    sPCA.components_.T,
    index=spread.columns,
    columns=["SPC1", "SPC2", "SPC3"],
)

cScore = pd.concat([tScore, sScore], axis=1).asfreq("MS")
pCScore = adfTest(cScore)
cScoreDiff = cScore.diff().dropna()
pCScoreDiff = adfTest(cScoreDiff)

model = VAR(cScoreDiff)
result = model.fit(maxlags=24, ic="aic", trend="c")
wTest = pd.Series(vars(result.test_whiteness()))

print(pCScore)
print(pCScoreDiff)
print(wTest)
print(result.summary())
print(result.resid)
print(result.sigma_u)

elements = [
    {
        "data": tLoading,
        "title": "Zero-Coupon Treasury Curve PCA Loadings",
        "xlabel": "Maturity",
        "ylabel": "Loading",
        "filename": "treasury_loadings.svg",
    },
    {
        "data": sLoading,
        "title": "Zero-Coupon Credit Spread Curve PCA Loadings",
        "xlabel": "Maturity",
        "ylabel": "Loading",
        "filename": "spread_loadings.svg",
    },
]

for e in elements:
    plot(**e)
plt.show()
