from pydantic import PositiveInt, validate_call
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from .esg import ESG
from .diagnostic import VAR_RULES, Severity

import numpy as np
import pandas as pd
import warnings


class ESGVar(ESG[VARResultsWrapper]):
    __slots__ = ()

    def validate(self) -> None:
        for rule in VAR_RULES:
            result = rule.condition(self.model)
            if result:
                message = rule.message
                if isinstance(result, tuple):
                    message = f"{message}; Failing columns - {result}"
                if rule.severity is Severity.ERROR:
                    raise ValueError(message)
                warnings.warn(message)

    def getHistory(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.endog[-self.model.k_ar :],
            self.model.model.data.row_labels[-self.model.k_ar :],
            self.model.names,
        )

    def getForecastDates(self, horizon: int) -> pd.Index:
        dates = self.model.model.data.dates
        return pd.Index(
            [dates[-1] + i * dates.freq for i in range(1, horizon + 1)], name="date"
        )

    @validate_call(config={"strict": True})
    def forecastMean(self, horizon: PositiveInt) -> pd.DataFrame:
        history = self.getHistory()
        mean = self.model.forecast(history.to_numpy(), horizon)
        return pd.DataFrame(
            mean,
            self.getForecastDates(horizon),
            self.model.names,
        )

    def step(self, history: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
        mean = self.model.forecast(history.to_numpy(), 1)[0]
        shock = rng.multivariate_normal(
            np.zeros(self.model.neqs),
            self.model.sigma_u,
            check_valid="raise",
        )
        return pd.Series(
            mean + shock,
            index=self.model.names,
            name=history.index[-1] + self.model.model.data.dates.freq,
        )
