from collections.abc import Callable
from pydantic import PositiveInt, validate_call
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG, Severity
from .diagnostic import *

import numpy as np
import pandas as pd
import warnings

Rule = tuple[Callable[[VARResultsWrapper], bool | tuple[int, ...]], Severity, str]
NLAGS = 24
STEP = 1
RULES: tuple[Rule, ...] = (
    (
        lambda m: m.model.data.dates is None or m.model.data.dates.freq is None,
        Severity.ERROR,
        "VAR requires dated observations with a regular frequency.",
    ),
    (lambda m: m.k_ar == 0, Severity.ERROR, "VAR has no lags"),
    (lambda m: not m.is_stable(), Severity.ERROR, "VAR is unstable"),
    (
        lambda m: adfTest(m.endog),
        Severity.WARNING,
        "ADF - could not reject a unit root in at least one factor",
    ),
    (
        lambda m: kpssTest(m.endog),
        Severity.WARNING,
        "KPSS - rejected level stationarity in at least one factor",
    ),
    (
        lambda m: m.test_normality(SIGNIF).conclusion == "reject",
        Severity.WARNING,
        "Normality - evidence against Gaussian residuals",
    ),
    (
        lambda m: m.test_whiteness(
            nlags=max(NLAGS, m.k_ar + 1), signif=SIGNIF
        ).conclusion
        == "reject",
        Severity.WARNING,
        "Whiteness - evidence of residual autocorrelation",
    ),
    (
        lambda m: archTest(m.resid.to_numpy()),
        Severity.WARNING,
        "ARCH - evidence of conditional heteroscedasticity in at least one residual series",
    ),
)


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def validate(self) -> None:
        for condition, severity, message in RULES:
            result = condition(self.model)
            if result:
                if isinstance(result, tuple):
                    message = f"{message}; Failing columns - {result}"
                if severity is Severity.ERROR:
                    raise ValueError(message)
                warnings.warn(message)

    def initHistory(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.endog[-self.model.k_ar :],
            columns=self.model.names,
            index=self.model.model.data.row_labels[-self.model.k_ar :],
        )

    @validate_call(config={"strict": True})
    def forecastMean(self, horizon: PositiveInt) -> pd.DataFrame:
        history = self.initHistory()
        mean = self.model.forecast(history.to_numpy(), horizon)
        dates = [
            history.index[-1] + (i + 1) * self.model.model.data.dates.freq
            for i in range(horizon)
        ]
        return pd.DataFrame(
            mean,
            columns=self.model.names,
            index=pd.Index(dates, name="date"),
        )

    def step(self, history: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
        mean = self.model.forecast(history.to_numpy(), STEP)[0]
        shock = rng.multivariate_normal(
            np.zeros(self.model.neqs),
            self.model.sigma_u,
            check_valid="raise",
        )
        return pd.Series(
            mean + shock,
            index=self.model.names,
            name=history.index[-1] + self.model.model.data.dates.freq,
        )
