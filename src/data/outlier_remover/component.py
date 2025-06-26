from typing import Optional, Tuple

import pandas as pd
from sklearn.exceptions import NotFittedError

from src.utils.logger import get_logger
from .schema import OutlierRemoverConfig

logger = get_logger(__name__)

class OutlierRemover:
    """
    Calculates group-wise statistics during 'fit' and uses them to filter
    outliers during 'transform'. This is a pre-processing step that
    filters rows from the dataset based on learned group means.
    """
    def __init__(self, config: OutlierRemoverConfig):
        """
        Initializes the DataCleaner with a configuration.

        Args:
            config (DataCleanerConfig): The configuration object that defines
                the cleaning parameters.
        """
        self.config = config
        logger.info(f"Initialized DataCleaner with config: {config.model_dump()}")

    def fit(self, X: pd.DataFrame):
        """
        Calculates the mean of the temperature_column for each group defined
        by group_by_column and stores it internally.

        Args:
            X (pd.DataFrame): The training dataframe to learn the group means from.

        Returns:
            self: The fitted instance of the DataCleaner.
        """
        logger.info(f"Fitting DataCleaner: calculating mean of '{self.config.temperature_column}' "
                    f"grouped by '{self.config.group_by_column}'.")
        
        # Validate that required columns exist
        required_cols = [self.config.group_by_column, self.config.temperature_column]
        if not all(col in X.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        self._group_means: pd.Series = X.groupby(self.config.group_by_column)[self.config.temperature_column].mean()
        logger.info(f"Fit complete. Found means for {len(self._group_means)} groups.")
        
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Filters the dataframe X (and optionally y) by removing rows belonging
        to groups whose learned mean temperature is outside the specified thresholds.

        Args:
            X (pd.DataFrame): The dataframe to transform.
            y (pd.Series, optional): The target series to transform, which will be
                aligned with the filtered X. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: A tuple containing the
            transformed X and the (optionally) transformed y.
        """
        if not hasattr(self, '_group_means'):
            raise NotFittedError("This DataCleaner instance is not fitted yet. Call 'fit' before transforming data.")

        logger.info("Transforming data: removing outliers based on fitted group means.")
        initial_rows = len(X)

        # Create a temporary column with the group means for filtering
        # We use a temporary name to avoid conflicts
        mean_col_name = f"_temp_mean_{self.config.group_by_column}"
        X_transformed = X.merge(
            self._group_means.rename(mean_col_name),
            left_on=self.config.group_by_column,
            right_index=True,
            how='left'
        )
        
        # For groups in the transform set that were not in the fit set,
        # we can either drop them or fill with a neutral value.
        # Here, we fill with a value that will pass the filter, but dropping is also valid.
        # A more robust approach might be to fill with the global mean.
        # For simplicity, we'll let them be NaN and the filter will handle them.
        
        # Define the filter mask
        mask = (
            (X_transformed[mean_col_name] >= self.config.min_temp_threshold) &
            (X_transformed[mean_col_name] <= self.config.max_temp_threshold)
        )
        
        # Apply the mask
        X_filtered = X_transformed[mask].drop(columns=[mean_col_name])
        
        rows_removed = initial_rows - len(X_filtered)
        logger.info(f"Removed {rows_removed} rows ({rows_removed / initial_rows:.2%} of total).")
        logger.info(f"Data shape before: {X.shape}, after: {X_filtered.shape}")

        if y is not None:
            # Align y with the filtered X using the index
            y_filtered = y.loc[X_filtered.index]
            return X_filtered, y_filtered
        
        return X_filtered, None

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        A convenience method that first fits the data and then transforms it.

        Args:
            X (pd.DataFrame): The dataframe to fit and transform.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: The transformed X and y.
        """
        self.fit(X)
        return self.transform(X, y)