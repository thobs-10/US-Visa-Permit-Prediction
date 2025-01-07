from abc import ABC, abstractmethod
import pandas as pd

class Component(ABC):

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, data: pd.DataFrame, filename: str) -> None:
        pass