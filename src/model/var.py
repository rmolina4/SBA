from collections.abc import Callable
from typing import NamedTuple
from pydantic import PositiveInt, validate_call
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG, Severity
from .diagnostic import *

import numpy as np
import pandas as pd
import warnings


class Rule(NamedTuple):
    condition: Callable[[VARResultsWrapper], bool | tuple[int, ...]]
    severity: Severity
    message: str


NLAGS = 24
STEP = 1
RULES: tuple[Rule, ...] = (
    Rule.__new__(
        Rule,
        condition=lambda m: m.model.data.dates is None
        or m.model.data.dates.freq is None,
        severity=Severity.ERROR,
        message="VAR requires dated observations with a regular frequency.",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: m.k_ar == 0,
        severity=Severity.ERROR,
        message="VAR has no lags",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: not m.is_stable(),
        severity=Severity.ERROR,
        message="VAR is unstable",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: adfTest(m.endog),
        severity=Severity.WARNING,
        message="ADF - could not reject a unit root in at least one factor",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: kpssTest(m.endog),
        severity=Severity.WARNING,
        message="KPSS - rejected level stationarity in at least one factor",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: m.test_normality(SIGNIF).conclusion == "reject",
        severity=Severity.WARNING,
        message="Normality - evidence against Gaussian residuals",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: m.test_whiteness(
            nlags=max(NLAGS, m.k_ar + 1), signif=SIGNIF
        ).conclusion
        == "reject",
        severity=Severity.WARNING,
        message="Whiteness - evidence of residual autocorrelation",
    ),
    Rule.__new__(
        Rule,
        condition=lambda m: archTest(m.resid.to_numpy()),
        severity=Severity.WARNING,
        message="ARCH - evidence of conditional heteroscedasticity in at least one residual series",
    ),
)


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def validate(self) -> None:
        for rule in RULES:
            result = rule.condition(self.model)
            if result:
                message = rule.message
                if isinstance(result, tuple):
                    message = f"{message}; Failing columns - {result}"
                if rule.severity is Severity.ERROR:
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
