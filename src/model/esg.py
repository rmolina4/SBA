from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pandas.tseries.offsets import BaseOffset
from pydantic import NonNegativeInt, PositiveInt, TypeAdapter, validate_call
from typing import Final, cast

from .diagnostic import Result, Rule, Severity, getLagOrder

import numpy as np
import pandas as pd
import warnings

DATES_ADAPTER: Final = TypeAdapter(
    pd.DatetimeIndex | pd.PeriodIndex,
    config={"arbitrary_types_allowed": True},
)
FREQUENCY_ADAPTER: Final = TypeAdapter(
    BaseOffset,
    config={"arbitrary_types_allowed": True},
)


@dataclass(frozen=True, slots=True)
class ESG[ResultT: Result](ABC):
    model: ResultT
    dates: pd.DatetimeIndex | pd.PeriodIndex = field(
        init=False, repr=False, compare=False
    )
    frequency: BaseOffset = field(init=False, repr=False, compare=False)
    lagOrder: int = field(init=False, repr=False, compare=False)
    history: pd.DataFrame = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()
        dates = DATES_ADAPTER.validate_python(self.model.dates)
        frequency = FREQUENCY_ADAPTER.validate_python(dates.freq)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "lagOrder", getLagOrder(self.model))
        object.__setattr__(self, "history", self.getHistory())

    def validate(self) -> None:
        for rule in self.getRules():
            result = rule.condition(self.model)
            if result:
                message = rule.message
                if isinstance(result, tuple):
                    message = f"{message}; Failing columns - {result}"
                if rule.severity is Severity.ERROR:
                    raise ValueError(message)
                warnings.warn(message)

    @staticmethod
    @abstractmethod
    def getRules() -> tuple[Rule[Result] | Rule[ResultT], ...]: ...

    @abstractmethod
    def getHistory(self) -> pd.DataFrame: ...

    @abstractmethod
    def forecastMean(
        self, horizon: PositiveInt, history: pd.DataFrame | None = None
    ) -> pd.DataFrame: ...

    def getShock(self, rng: np.random.Generator) -> pd.Series:
        return pd.Series(
            rng.multivariate_normal(
                np.zeros(self.history.shape[1]),
                cast(np.ndarray, self.model.sigma_u),
                check_valid="raise",
            ),
            index=self.history.columns,
        )

    def step(self, history: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
        mean = self.forecastMean(1, history).iloc[0]
        return (mean + self.getShock(rng)).rename(history.index[-1] + self.frequency)

    def getForecastDates(
        self, horizon: int, history: pd.DataFrame | None = None
    ) -> pd.Index:
        dates = self.dates if history is None else history.index
        return pd.Index(
            [dates[-1] + i * self.frequency for i in range(1, horizon + 1)],
            name="date",
        )

    @validate_call(config={"strict": True})
    def generateForecast(
        self,
        horizon: PositiveInt = 60,
        scenario: PositiveInt = 1000,
        seed: NonNegativeInt = 0,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        scenarios: list[pd.DataFrame] = []
        for _ in range(scenario):
            history = self.history.copy()
            for _ in range(horizon):
                row = self.step(history, rng)
                history.loc[row.name] = row
            scenarios.append(history.tail(horizon))
        return pd.concat(scenarios, keys=range(scenario), names=["scenario", "date"])
