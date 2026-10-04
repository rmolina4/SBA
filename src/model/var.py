from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG

import numpy as np
import pandas as pd


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def validate(self) -> None:
        raise NotImplementedError("Not implemented yet.")

    def step(self, df: pd.DataFrame, it: int, rng: np.random.Generator) -> None:
        raise NotImplementedError("Not implemented yet.")
