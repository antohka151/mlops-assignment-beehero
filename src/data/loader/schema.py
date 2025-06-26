from pydantic import BaseModel, FilePath
from typing import Literal

class DataLoaderConfig(BaseModel):
    """
    Configuration schema for the DataLoader.

    Attributes:
        type: The type of the data source. Currently only 'csv' is supported.
        path: The path to the data file.
    """
    type: Literal["csv"]
    path: FilePath
