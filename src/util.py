from collections.abc import Callable
from typing import cast
from pydantic import PositiveInt, validate_call
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA

from model.var import ESGVar

import pandas as pd
import warnings

MAX_LAGS = 24
TCOLUMNS = ["TPC1", "TPC2", "TPC3"]
SCOLUMNS = ["SPC1", "SPC2", "SPC3"]
WARNINGS = True


def runPCA(
    df: pd.DataFrame, columns: list[str]
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    pca = PCA(3)
    score = pd.DataFrame(
        pca.fit_transform(df),
        index=df.index,
        columns=columns,
    )
    loading = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=columns,
    )
    return pca, score, loading


def runIPCA(
    treasury: pd.DataFrame, spread: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, PCA, PCA]:
    tPCA, tScore, tLoading = runPCA(treasury, TCOLUMNS)
    sPCA, sScore, sLoading = runPCA(spread, SCOLUMNS)
    cScore = pd.concat([tScore, sScore], axis=1).asfreq("MS").dropna()
    cLoading = pd.concat([tLoading, sLoading], axis=1)
    return (
        cScore,
        cLoading,
        tPCA,
        sPCA,
    )


def createESG(df: pd.DataFrame) -> ESGVar:
    esgWarnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as esgWarnings:
        warnings.simplefilter("always")
        fit = cast(Callable[..., VARResultsWrapper], VAR(df).fit)
        esg = ESGVar(fit(MAX_LAGS, ic="aic", trend="c"))
    if WARNINGS:
        print("\nESG:")
        for w in esgWarnings:
            print(f"\t{w.message}")
    return esg


def forecastIMean(
    esg: ESGVar,
    cScore: pd.DataFrame,
    tPCA: PCA,
    sPCA: PCA,
    maturities: pd.Index,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = esg.forecastMean(horizon).cumsum() + cScore.iloc[-1]
    tMean = pd.DataFrame(
        tPCA.inverse_transform(mean[TCOLUMNS]),
        index=mean.index,
        columns=maturities,
    )
    sMean = pd.DataFrame(
        sPCA.inverse_transform(mean[SCOLUMNS]),
        index=mean.index,
        columns=maturities,
    )
    cMean = tMean + sMean
    return tMean, cMean


def getHistory(
    treasury: pd.DataFrame, spread: pd.DataFrame, esg: ESGVar, cScore: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = cScore.index[-esg.model.k_ar :]
    tHistory = treasury.loc[dates]
    cHistory = tHistory + spread.loc[dates]
    return tHistory, cHistory


@validate_call(config={"strict": True, "arbitrary_types_allowed": True})
def forecastInterest(
    treasury: pd.DataFrame,
    spread: pd.DataFrame,
    horizon: PositiveInt = 60,
) -> tuple[pd.DataFrame, ...]:
    cScore, cLoading, tPCA, sPCA = runIPCA(treasury, spread)
    cScoreDiff = cScore.diff().dropna()
    esg = createESG(cScoreDiff)
    maturities = treasury.columns

    forecast = (
        esg.generateForecast(horizon).groupby(level="scenario").cumsum()
        + cScore.iloc[-1]
    )
    tForecast = pd.DataFrame(
        tPCA.inverse_transform(forecast[TCOLUMNS]),
        index=forecast.index,
        columns=maturities,
    )
    sForecast = pd.DataFrame(
        sPCA.inverse_transform(forecast[SCOLUMNS]),
        index=forecast.index,
        columns=maturities,
    )
    cForecast = tForecast + sForecast

    tMean, cMean = forecastIMean(esg, cScore, tPCA, sPCA, maturities, horizon)
    tHistory, cHistory = getHistory(treasury, spread, esg, cScore)
    return (
        cScore,
        cScoreDiff,
        cLoading,
        tForecast,
        cForecast,
        tMean,
        cMean,
        tHistory,
        cHistory,
    )
