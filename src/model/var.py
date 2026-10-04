from collections.abc import Callable
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG, Severity

import numpy as np
import pandas as pd
import warnings

SIGNIFICANCE_LEVEL = 0.05


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def validate(self) -> None:
        rules: list[tuple[Callable[[VARResultsWrapper], bool], Severity, str]] = [
            (lambda m: m.k_ar == 0, Severity.ERROR, "VAR has no lags"),
            (lambda m: not m.is_stable(), Severity.ERROR, "VAR is unstable"),
            (
                lambda m: m.test_whiteness(
                    nlags=max(24, m.k_ar + 1), signif=SIGNIFICANCE_LEVEL
                ).pvalue < SIGNIFICANCE_LEVEL,
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
