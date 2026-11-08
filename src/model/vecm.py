from pydantic import PositiveInt, validate_call
from statsmodels.tsa.vector_ar.var_model import (
    forecast,  # pyright: ignore[reportUnknownVariableType]
)
from statsmodels.tsa.vector_ar.vecm import VECMResults
from typing import cast

from .esg import ESG
from .diagnostic import VECM_RULES, Result, Rule

import numpy as np
import pandas as pd


class ESGVecm(ESG[VECMResults]):
    __slots__ = ()

    @staticmethod
    def getRules() -> tuple[Rule[Result] | Rule[VECMResults], ...]:
        return VECM_RULES

    def getHistory(self) -> pd.DataFrame:
        return pd.DataFrame(
            cast(np.ndarray, self.model.y_all).T[-self.lagOrder :],
            self.dates[-self.lagOrder :],
            cast(list[str], self.model.names),
        )

    @validate_call(config={"strict": True, "arbitrary_types_allowed": True})
    def forecastMean(
        self, horizon: PositiveInt, history: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        history = self.history if history is None else history
        return pd.DataFrame(
            forecast(
                history.to_numpy(),
                self.model.var_rep,
                cast(np.ndarray, self.model.const).T,
                horizon,
            ),
            self.getForecastDates(horizon, history),
            cast(list[str], self.model.names),
        )
