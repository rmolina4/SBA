from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pandas.tseries.offsets import BaseOffset
from pydantic import NonNegativeInt, PositiveInt, TypeAdapter, validate_call

from .diagnostic import Result, Rule, Severity, getLagOrder

import numpy as np
import pandas as pd
import warnings

DATES_ADAPTER = TypeAdapter(
    pd.DatetimeIndex | pd.PeriodIndex,
    config={"arbitrary_types_allowed": True},
)
FREQUENCY_ADAPTER = TypeAdapter(
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
    k_ar: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dates = DATES_ADAPTER.validate_python(self.model.dates)
        frequency = FREQUENCY_ADAPTER.validate_python(dates.freq)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "k_ar", getLagOrder(self.model))
        self.validate()

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

    @abstractmethod
    def getRules(self) -> tuple[Rule[Result] | Rule[ResultT], ...]: ...

    @abstractmethod
    def getHistory(self) -> pd.DataFrame: ...

    @abstractmethod
    def step(self, history: pd.DataFrame, rng: np.random.Generator) -> pd.Series: ...

    @abstractmethod
    def forecastMean(self, horizon: int) -> pd.DataFrame: ...

    def getForecastDates(self, horizon: int) -> pd.Index:
        return pd.Index(
            [self.dates[-1] + i * self.frequency for i in range(1, horizon + 1)],
            name="date",
        )

    @validate_call(config={"strict": True})
    def generateForecast(
        self,
        horizon: PositiveInt = 60,
        scenario: PositiveInt = 10000,
        seed: NonNegativeInt = 0,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        initial = self.getHistory()
        scenarios: list[pd.DataFrame] = []
        for _ in range(scenario):
            history = initial.copy()
            for _ in range(horizon):
                row = self.step(history, rng)
                history.loc[row.name] = row
            scenarios.append(history.tail(horizon))
        return pd.concat(scenarios, keys=range(scenario), names=["scenario", "date"])
