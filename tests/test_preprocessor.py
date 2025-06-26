import pytest
import pandas as pd
from src.data.preprocessor.component import FeaturesPreprocessor
from src.data.preprocessor.schema import PreprocessorStepConfig

# Minimal configs for each transformer
def get_steps():
    return [
        PreprocessorStepConfig(
            name="int_to_float",
            class_path="feature_store.IntToFloatConverter",
            params={"columns": ["int_col"]}
        ),
        PreprocessorStepConfig(
            name="mean_imputer",
            class_path="feature_store.MeanImputer",
            params={"input_cols": ["float_col"]}
        ),
        PreprocessorStepConfig(
            name="column_subtractor",
            class_path="feature_store.ColumnSubtractor",
            params={
                "input_cols": ["float_col", "int_col"],
                "output_col": "diff_col",
                "col_a": "float_col",
                "col_b": "int_col"
            }
        ),
        PreprocessorStepConfig(
            name="threshold_binarizer",
            class_path="feature_store.ThresholdBinarizer",
            params={
                "input_col": "float_col",
                "output_col": "bin_col",
                "threshold": 1.5
            }
        ),
    ]

def get_df():
    return pd.DataFrame({
        "int_col": [1, 2, 3, 4],
        "float_col": [1.0, 2.0, None, 4.0],
    })

def test_fit_transform_pipeline():
    steps = get_steps()
    df = get_df()
    preprocessor = FeaturesPreprocessor(steps)
    preprocessor.fit(df)
    result = preprocessor.transform(df)
    # Check columns
    assert "diff_col" in result.columns
    assert "bin_col" in result.columns
    # Check int_col is float
    assert result["int_col"].dtype == "float64"
    # Check imputation
    assert not result["float_col"].isnull().any()
    # Check binarizer
    assert set(result["bin_col"].unique()).issubset({0, 1})

def test_metadata():
    steps = get_steps()
    df = get_df()
    preprocessor = FeaturesPreprocessor(steps)
    preprocessor.fit(df)
    metadata = preprocessor.get_metadata()
    assert isinstance(metadata, list)
    assert all("name" in m.model_dump() for m in metadata)
    assert all("class_path" in m.model_dump() for m in metadata)

def test_error_on_missing_column():
    steps = get_steps()
    df = pd.DataFrame({"float_col": [1.0, 2.0, 3.0, 4.0]})
    preprocessor = FeaturesPreprocessor(steps)
    with pytest.raises(ValueError):
        preprocessor.fit(df)

def test_error_on_transform_before_fit():
    steps = get_steps()
    df = get_df()
    preprocessor = FeaturesPreprocessor(steps)
    with pytest.raises(RuntimeError):
        preprocessor.transform(df)
