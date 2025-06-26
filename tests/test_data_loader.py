import os
import tempfile
import pandas as pd
import pytest
from src.data.loader.component import DataLoader
from src.data.loader.schema import DataLoaderConfig


def test_load_from_csv_success():
    # Create a temporary CSV file
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    try:
        config = DataLoaderConfig(type="csv", path=tmp_path)
        loader = DataLoader(config)
        loaded_df = loader.load_data()
        pd.testing.assert_frame_equal(loaded_df, df)
    finally:
        os.remove(tmp_path)

def test_load_from_csv_file_not_found():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        DataLoaderConfig(type="csv", path="/non/existent/file.csv")

def test_unsupported_data_source_type():
    class DummyConfig:
        type = "json"
        path = "dummy.json"
    config = DummyConfig()
    loader = DataLoader(config)
    with pytest.raises(ValueError) as excinfo:
        loader.load_data()
    assert "Unsupported data source type" in str(excinfo.value)
