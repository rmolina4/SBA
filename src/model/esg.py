from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

ModelT = TypeVar("ModelT")


class ESG(ABC, Generic[ModelT]):
    __slots__ = ("model",)

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
    def step(self, df: pd.DataFrame, it: int, rng: np.random.Generator) -> None: ...

    def generateForecast(
        self,
        df: pd.DataFrame,
        horizon: int = 60,
        seed: int = 0,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        for it in range(horizon):
            self.step(df, it, rng)
        return pd.DataFrame()
