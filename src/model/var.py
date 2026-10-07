from collections.abc import Callable
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG, Severity
from .diagnostic import *

import numpy as np
import pandas as pd
import warnings

Rule = tuple[Callable[[VARResultsWrapper], bool], Severity, str]
NLAGS = 24
RULES: tuple[Rule, ...] = (
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
        lambda m: johansenTest(m.endog, m.k_ar),
        Severity.WARNING,
        "Johansen - no cointegrating relationship detected",
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
            if condition(self.model):
                if severity is Severity.ERROR:
                    raise ValueError(message)
                warnings.warn(message)

    def step(self, df: pd.DataFrame, it: int, rng: np.random.Generator) -> None:
        raise NotImplementedError("Not implemented yet.")
