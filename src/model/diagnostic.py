from enum import Enum
from statsmodels.stats.diagnostic import (
    het_arch,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.stattools import (
    adfuller,  # pyright: ignore[reportUnknownVariableType]
    kpss,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.vector_ar.vecm import (
    select_coint_rank,  # pyright: ignore[reportUnknownVariableType]
)

import numpy as np

SIGNIF = 0.05
CointegrationStatus = Enum("CointegrationStatus", ["NONE", "COINTEGRATED", "FULL_RANK"])


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
