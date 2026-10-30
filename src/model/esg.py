from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import NonNegativeInt, PositiveInt, validate_call

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ESG[ResultT](ABC):
    model: ResultT

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def validate(self) -> None: ...

    @abstractmethod
    def getHistory(self) -> pd.DataFrame: ...

    @abstractmethod
    def step(self, history: pd.DataFrame, rng: np.random.Generator) -> pd.Series: ...

    @abstractmethod
    def forecastMean(self, horizon: int) -> pd.DataFrame: ...

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
