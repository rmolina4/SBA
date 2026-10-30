from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.stats.diagnostic import (
    het_arch,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.stattools import (
    adfuller,  # pyright: ignore[reportUnknownVariableType]
    kpss,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.vector_ar.vecm import (
    VECMResults,
    select_coint_rank,  # pyright: ignore[reportUnknownVariableType]
)

import numpy as np

MAX_LAGS = 24
SIGNIF = 0.05
Severity = Enum("Severity", ["ERROR", "WARNING"])
CointegrationStatus = Enum("CointegrationStatus", ["NONE", "COINTEGRATED", "FULL_RANK"])
type Result = VARResultsWrapper | VECMResults


@dataclass(frozen=True, slots=True)
class Rule[ResultT: Result]:
    condition: Callable[[ResultT], bool | tuple[int, ...]]
    severity: Severity
    message: str


BASE_RULES: tuple[Rule[Result], ...] = (
    Rule[Result](
        lambda m: m.dates is None or m.dates.freq is None,
        Severity.ERROR,
        "Model requires dated observations with a regular frequency.",
    ),
    Rule[Result](
        lambda m: cast(int, m.k_ar) <= 0,
        Severity.ERROR,
        "Model has no lags",
    ),
)


RESIDUAL_RULES: tuple[Rule[Result], ...] = (
    Rule[Result](
        lambda m: m.test_normality(SIGNIF).conclusion == "reject",
        Severity.WARNING,
        "Normality - evidence against Gaussian residuals",
    ),
    Rule[Result](
        lambda m: m.test_whiteness(
            max(MAX_LAGS, cast(int, m.k_ar) + 1), SIGNIF
        ).conclusion
        == "reject",
        Severity.WARNING,
        "Whiteness - evidence of residual autocorrelation",
    ),
)


VAR_RULES: tuple[Rule[Result] | Rule[VARResultsWrapper], ...] = (
    *BASE_RULES,
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
    Rule[VARResultsWrapper](
        lambda m: archTest(m.resid.to_numpy()),
        Severity.WARNING,
        "ARCH - evidence of conditional heteroscedasticity in at least one residual series",
    ),
)


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


def johansenTest(data: np.ndarray, k_ar: int) -> tuple[int, CointegrationStatus]:
    result = select_coint_rank(
        data,
        0,
        k_ar - 1,
        signif=SIGNIF,
    )
    if result.rank == 0:
        status = CointegrationStatus.NONE
    elif result.rank == result.neqs:
        status = CointegrationStatus.FULL_RANK
    else:
        status = CointegrationStatus.COINTEGRATED
    return (result.rank, status)
