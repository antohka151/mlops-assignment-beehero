from typing import Protocol
import pandas as pd
from src.utils.logger import get_logger
from .schema import DataLoaderConfig

logger = get_logger(__name__)

# Interface for loader functions
class LoaderStrategy(Protocol):
    def __call__(self, path: str) -> pd.DataFrame:
        ...

# --- Concrete Strategies ---
def _load_from_csv(path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    logger.info(f"Loading data from CSV at {path}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"File not found at path: {path}")
        raise 

class DataLoader:
    """Handles loading data from various sources using a strategy pattern."""
    _STRATEGIES: dict[str, LoaderStrategy] = {
        "csv": _load_from_csv,
    }

    def __init__(self, config: DataLoaderConfig):
        """Initializes the DataLoader."""
        self.config = config
        logger.info("DataLoader initialized.")

    def load_data(self) -> pd.DataFrame:
        """
        Loads data by dispatching to the correct strategy based on config.
        
        Returns:
            pd.DataFrame: The loaded data.
        Raises:
            ValueError: If the source type is unsupported.
        """
        source_type = self.config.type
        path = self.config.path

        strategy = self._STRATEGIES.get(source_type)
        if strategy is None:
            error_msg = (
                f"Unsupported data source type: {source_type}.\n"
                f"Supported types are: {list(self._STRATEGIES.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return strategy(str(path))