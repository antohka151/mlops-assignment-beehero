from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class that defines the contract for all
    feature transformers in the pipeline. It is scikit-learn compatible.
    """
    def _validate_columns(self, df: pd.DataFrame, required_cols: list[str]):
        """Helper to check for presence of required columns."""
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"'{self.__class__.__name__}' is missing required columns: {missing}")

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns parameters from the data. Must return self.
        The 'y=None' is for compatibility with sklearn pipelines but is not used.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned transformation to X.
        """
        raise NotImplementedError

class IntToFloatConverter(FeatureTransformer):
    """
    Converts specified integer columns to float64 type.
    This is useful for making model signatures more robust to potential
    missing values at inference time.
    """
    def __init__(self, columns: list[str] = None):
        """
        Args:
            columns (list[str], optional): A list of columns to convert.
                                           If None, all integer columns will be converted.
                                           Defaults to None.
        """
        self.columns = columns
        self.columns_to_convert_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Identifies the integer columns to be converted.
        """
        if self.columns:
            # Use user-specified columns
            self._validate_columns(X, self.columns)
            self.columns_to_convert_ = self.columns
        else:
            # Automatically find all integer columns
            self.columns_to_convert_ = X.select_dtypes(
                include=['int', 'int32', 'int64']
            ).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the type conversion to the identified columns.
        """
        X_out = X.copy()
        if not self.columns_to_convert_:
            # This can happen if fit was called on data with no integers
            # and no columns were specified.
            return X_out

        for col in self.columns_to_convert_:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype('float64')
        return X_out

class MeanImputer(FeatureTransformer):
    """Fills missing values in specified columns with their learned means."""
    def __init__(self, input_cols: list[str]):
        self.input_cols = input_cols
        self.imputation_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X, self.input_cols)
        for col in self.input_cols:
            self.imputation_values_[col] = X[col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        return X_out.fillna(self.imputation_values_)

class GroupedAggregator(FeatureTransformer):
    """Calculates aggregations and merges them back into the DataFrame."""
    def __init__(self, input_cols: list[str], groupby_col: str, aggregations: dict[str, list[str]], output_cols: list[str]):
        self.input_cols = input_cols
        self.groupby_col = groupby_col
        self.aggregations = aggregations
        self.output_cols = output_cols
        self.agg_df_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X, self.input_cols + [self.groupby_col])
        agg_cols = list(self.aggregations.keys())
        self.agg_df_ = X.groupby(self.groupby_col)[agg_cols].agg(self.aggregations)
        self.agg_df_.columns = self.output_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X, [self.groupby_col])
        if any(col in X.columns for col in self.output_cols):
            raise ValueError(f"'{self.__class__.__name__}' would overwrite existing columns: {self.output_cols}")
        
        X_out = X.merge(
            self.agg_df_, 
            left_on=self.groupby_col, 
            right_index=True, 
            how='left'
        )
        return X_out

class ColumnSubtractor(FeatureTransformer):
    """Creates a new column by subtracting one from another."""
    def __init__(self, input_cols: list[str], output_col: str, col_a: str, col_b: str):
        self.input_cols = input_cols
        self.output_col = output_col
        self.col_a = col_a
        self.col_b = col_b

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X, self.input_cols)
        return self # Stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X, self.input_cols)
        if self.output_col in X.columns:
            raise ValueError(f"'{self.__class__.__name__}' would overwrite existing columns: {self.output_col}")

        X_out = X.copy()
        X_out[self.output_col] = X_out[self.col_a] - X_out[self.col_b]
        return X_out

class AbsoluteDifference(FeatureTransformer):
    """Creates a new column with the absolute difference between two columns."""
    def __init__(self, input_cols: list[str], output_col: str, col_a: str, col_b: str):
        self.input_cols = input_cols
        self.output_col = output_col
        self.col_a = col_a
        self.col_b = col_b

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X, self.input_cols)
        return self # Stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X, self.input_cols)
        if self.output_col in X.columns:
            raise ValueError(f"'{self.__class__.__name__}' would overwrite existing columns: {self.output_col}")
            
        X_out = X.copy()
        X_out[self.output_col] = abs(X_out[self.col_a] - X_out[self.col_b])
        return X_out

class ThresholdBinarizer(FeatureTransformer):
    """Creates a binary flag (0/1) if a column value is greater than a threshold."""
    def __init__(self, input_col: str, output_col: str, threshold: float):
        self.input_col = input_col
        self.output_col = output_col
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X, [self.input_col])
        return self # Stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X, [self.input_col])
        if self.output_col in X.columns:
            raise ValueError(f"'{self.__class__.__name__}' would overwrite existing columns: {self.output_col}")

        X_out = X.copy()
        X_out[self.output_col] = (X_out[self.input_col] > self.threshold).astype(int)
        return X_out