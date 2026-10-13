from statsmodels.stats.diagnostic import (
    het_arch,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.stattools import (
    adfuller,  # pyright: ignore[reportUnknownVariableType]
    kpss,  # pyright: ignore[reportUnknownVariableType]
)

import numpy as np

SIGNIF = 0.05


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
