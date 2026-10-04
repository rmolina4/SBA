from collections.abc import Callable
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.stattools import (
    adfuller,  # pyright: ignore[reportUnknownVariableType]
)

from .esg import ESG, Severity

import numpy as np
import pandas as pd
import warnings

SIGNIF = 0.05
NLAGS = 24


# null: unit root, assumes data is non-stationary
def adfTest(a: np.ndarray) -> bool:
    for j in range(a.shape[1]):
        if adfuller(a[:, j], autolag="AIC", result_object=True)[1] >= SIGNIF:
            return True
    return False


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def validate(self) -> None:
        rules: list[tuple[Callable[[VARResultsWrapper], bool], Severity, str]] = [
            (lambda m: m.k_ar == 0, Severity.ERROR, "VAR has no lags"),
            (lambda m: not m.is_stable(), Severity.ERROR, "VAR is unstable"),
            (lambda m: adfTest(m.endog), Severity.WARNING, "Failed ADF test"),
            (
                lambda m: m.test_normality(SIGNIF).conclusion
                == "reject",  # null: guassian errors
                Severity.WARNING,
                "Failed normality test",
            ),
            (
                lambda m: m.test_whiteness(
                    nlags=max(NLAGS, m.k_ar + 1), signif=SIGNIF
                ).conclusion
                == "reject",  # null: no residual autocorrelation
                Severity.WARNING,
                "Failed Whiteness Test",
            ),
        ]
        for condition, severity, message in rules:
            if condition(self.model):
                if severity is Severity.ERROR:
                    raise ValueError(f"{self.name}: {message}")
                warnings.warn(f"{self.name}: {message}")

    def step(self, df: pd.DataFrame, it: int, rng: np.random.Generator) -> None:
        raise NotImplementedError("Not implemented yet.")
