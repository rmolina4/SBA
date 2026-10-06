from statsmodels.tsa.vector_ar.vecm import (
    select_coint_rank,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.stats.diagnostic import (
    het_arch,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.stattools import (
    adfuller,  # pyright: ignore[reportUnknownVariableType]
    kpss,  # pyright: ignore[reportUnknownVariableType]
)

import numpy as np

SIGNIF = 0.05


def adfTest(data: np.ndarray) -> bool:
    return any(
        adfuller(data[:, j], result_object=True)[1] >= SIGNIF
        for j in range(data.shape[1])
    )


def kpssTest(data: np.ndarray) -> bool:
    return any(
        kpss(data[:, j], result_object=True)[1] < SIGNIF for j in range(data.shape[1])
    )


def johansenTest(data: np.ndarray, k_ar: int) -> bool:
    k_ar_diff = k_ar - 1
    if data.shape[1] < 2:
        return False
    return select_coint_rank(data, det_order=0, k_ar_diff=k_ar_diff).rank == 0


def archTest(residuals: np.ndarray) -> bool:
    return any(
        het_arch(residuals[:, j], result_object=True)[1] < SIGNIF
        for j in range(residuals.shape[1])
    )
