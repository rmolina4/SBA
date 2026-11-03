from statsmodels.tsa.vector_ar.vecm import VECMResults
from typing import cast

from .esg import ESG
from .diagnostic import VECM_RULES, Result, Rule

import numpy as np
import pandas as pd


class ESGVecm(ESG[VECMResults]):
    __slots__ = ()

    def getRules(self) -> tuple[Rule[Result] | Rule[VECMResults], ...]:
        return VECM_RULES

    def getHistory(self) -> pd.DataFrame:
        return pd.DataFrame(
            cast(np.ndarray, self.model.y_all).T[-self.lagOrder :].copy(),
            self.dates[-self.lagOrder :],
            cast(list[str], self.model.names),
        )
