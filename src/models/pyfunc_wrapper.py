# src/models/pyfunc_wrapper.py

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.outlier_remover import OutlierRemover

class FullPipelinePyFunc(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for the complete prediction pipeline.
    This wrapper ensures that the index of the input data is preserved in the
    output predictions, allowing for correct alignment and traceability.
    """
    def __init__(self, outlier_remover: OutlierRemover, sklearn_pipeline: Pipeline):
        """
        Initializes the wrapper with pre-fitted components.

        Args:
            outlier_remover (OutlierRemover): A FITTED instance of the outlier remover.
            sklearn_pipeline (Pipeline): A FITTED instance of the main scikit-learn pipeline.
        """
        self._outlier_remover = outlier_remover
        # THE FIX: Directly assign the pre-fitted pipeline object.
        # Do not clone it or reconstruct it, as that would lose its fitted state.
        self._sklearn_pipeline = sklearn_pipeline

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        """
        Generates predictions from raw input data, returning a pandas Series
        with the original index preserved for the rows that were not filtered out.

        Args:
            context: MLflow context (unused here).
            model_input (pd.DataFrame): Raw input data with an index.

        Returns:
            pd.Series: A Series containing the predictions, indexed by the
                       original index of the input rows that were scored.
        """
        # Step 1: Apply the outlier remover.
        cleaned_x, _ = self._outlier_remover.transform(model_input)

        # If all rows were removed, return an empty Series.
        if cleaned_x.empty:
            # It's good practice to define the dtype of the empty series.
            # We can get it from the learned classes of the classifier.
            classifier = self._sklearn_pipeline['classifier']
            output_dtype = classifier.classes_.dtype
            return pd.Series(dtype=output_dtype)

        # The index of `cleaned_x` tells us which original rows survived filtering.
        final_index = cleaned_x.index

        # Step 2: Apply the main scikit-learn pipeline to the cleaned data.
        # This call will now work without warnings because self._sklearn_pipeline is the fitted instance.
        predictions_array = self._sklearn_pipeline.predict(cleaned_x)

        # Step 3: Combine the NumPy predictions with the preserved index.
        return pd.Series(predictions_array, index=final_index, name="prediction")