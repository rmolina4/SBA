from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from enum import Enum
from pydantic import NonNegativeInt, PositiveInt, validate_call

import numpy as np
import pandas as pd

ModelT = TypeVar("ModelT")
Severity = Enum("Severity", ["ERROR", "WARNING"])


class ESG(ABC, Generic[ModelT]):
    __slots__ = "model"

    def __init__(self, model: ModelT, check: bool = True) -> None:
        object.__setattr__(self, "model", model)
        self.validate() if check else None

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"Cannot modify immutable field '{name}'")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Cannot delete immutable field '{name}'")

    @abstractmethod
    def validate(self) -> None: ...

    @abstractmethod
    def initHistory(self) -> pd.DataFrame: ...

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
        init = self.initHistory()
        scenarios: list[pd.DataFrame] = []
        for _ in range(scenario):
            history = init.copy()
            for _ in range(horizon):
                row = self.step(history, rng)
                history.loc[row.name] = row
            scenarios.append(history.iloc[len(init) :])
        return pd.concat(
            scenarios,
            keys=range(scenario),
            names=["scenario", "date"],
        )
