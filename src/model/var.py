from pydantic import PositiveInt, validate_call
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG
from .diagnostic import VAR_RULES, Result, Rule

import pandas as pd


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def getRules(self) -> tuple[Rule[Result] | Rule[VARResultsWrapper], ...]:
        return VAR_RULES

    def getHistory(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.endog[-self.lagOrder :],
            self.dates[-self.lagOrder :],
            self.model.names,
        )

    @validate_call(config={"strict": True, "arbitrary_types_allowed": True})
    def forecastMean(
        self, horizon: PositiveInt, history: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        history = self.history if history is None else history
        return pd.DataFrame(
            self.model.forecast(history.to_numpy(), horizon),
            self.getForecastDates(horizon, history),
            self.model.names,
        )
