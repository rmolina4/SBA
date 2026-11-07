from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, cast
from numpy.typing import ArrayLike
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.stats.diagnostic import (
    het_arch,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.stattools import (
    adfuller,  # pyright: ignore[reportUnknownVariableType]
    kpss,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.vector_ar.vecm import VECMResults

import numpy as np

MAX_LAGS: Final = 24
SIGNIF: Final = 0.05
Severity = Enum("Severity", ["ERROR", "WARNING"])
type Result = VARResultsWrapper | VECMResults


@dataclass(frozen=True, slots=True)
class Rule[ResultT: Result]:
    condition: Callable[[ResultT], bool | tuple[int, ...]]
    severity: Severity
    message: str


BASE_RULES: Final[tuple[Rule[Result], ...]] = (
    Rule[Result](
        lambda m: m.exog is not None
        or (isinstance(m, VECMResults) and m.exog_coint is not None),
        Severity.ERROR,
        "Model has external regressors",
    ),
)


RESIDUAL_RULES: Final[tuple[Rule[Result], ...]] = (
    Rule[Result](
        lambda m: m.test_normality(SIGNIF).conclusion == "reject",
        Severity.WARNING,
        "Normality - evidence against Gaussian residuals",
    ),
    Rule[Result](
        lambda m: m.test_whiteness(max(MAX_LAGS, getLagOrder(m) + 1), SIGNIF).conclusion
        == "reject",
        Severity.WARNING,
        "Whiteness - evidence of residual autocorrelation",
    ),
    Rule[Result](
        lambda m: archTest(np.asarray(cast(ArrayLike, m.resid))),
        Severity.WARNING,
        "ARCH - evidence of conditional heteroscedasticity in at least one residual series",
    ),
)


VAR_RULES: Final[tuple[Rule[Result] | Rule[VARResultsWrapper], ...]] = (
    *BASE_RULES,
    Rule[VARResultsWrapper](
        lambda m: getLagOrder(m) <= 0,
        Severity.ERROR,
        "Model has no lags",
    ),
    Rule[VARResultsWrapper](
        lambda m: m.trend not in ("n", "c"),
        Severity.ERROR,
        "VAR supports only no trend or a constant",
    ),
    Rule[VARResultsWrapper](
        lambda m: not m.is_stable(),
        Severity.ERROR,
        "VAR is unstable",
    ),
    Rule[VARResultsWrapper](
        lambda m: adfTest(m.endog),
        Severity.WARNING,
        "ADF - could not reject a unit root in at least one factor",
    ),
    Rule[VARResultsWrapper](
        lambda m: kpssTest(m.endog),
        Severity.WARNING,
        "KPSS - rejected level stationarity in at least one factor",
    ),
    *RESIDUAL_RULES,
)


VECM_RULES: Final[tuple[Rule[Result] | Rule[VECMResults], ...]] = (
    *BASE_RULES,
    Rule[VECMResults](
        lambda m: cast(str, m.deterministic) != "co",
        Severity.ERROR,
        "VECM requires only a constant outside cointegration",
    ),
    Rule[VECMResults](
        lambda m: m.seasons != 0,
        Severity.ERROR,
        "VECM does not support seasonality",
    ),
    Rule[VECMResults](
        lambda m: not 0 < m.coint_rank < m.neqs,
        Severity.ERROR,
        "VECM requires a fitted rank with partial cointegration",
    ),
    *RESIDUAL_RULES,
)


def getLagOrder(model: Result) -> int:
    return cast(int, model.k_ar)


def adfTest(data: np.ndarray) -> tuple[int, ...]:
    return tuple(
        j
        for j in range(data.shape[1])
        if adfuller(data[:, j], result_object=True)[1] >= SIGNIF
    )


def kpssTest(data: np.ndarray) -> tuple[int, ...]:
    return tuple(
        j
        for j in range(data.shape[1])
        if kpss(data[:, j], result_object=True)[1] < SIGNIF
    )


def archTest(residuals: np.ndarray) -> tuple[int, ...]:
    return tuple(
        j
        for j in range(residuals.shape[1])
        if het_arch(residuals[:, j], result_object=True)[1] < SIGNIF
    )
