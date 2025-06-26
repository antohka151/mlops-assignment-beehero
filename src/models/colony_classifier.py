from typing import Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from src.utils.logger import get_logger
from pydantic import BaseModel, Field
import importlib
import pandas as pd

logger = get_logger(__name__)

class ColonyStrengthClassifierConfig(BaseModel):
    """Configuration for the model algorithm itself."""
    model_class_path: str = Field(default="sklearn.ensemble.RandomForestClassifier")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

class ColonyStrengthClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, config: ColonyStrengthClassifierConfig):
        self.config = config
        # Do NOT instantiate the model here. It will be created in fit().
        self._model = None
        self.classes_ = None

    def _create_model(self) -> BaseEstimator:
        """Dynamically imports and instantiates the model from its full class path."""
        try:
            module_path, class_name = self.config.model_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class(**self.config.hyperparameters)
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(
                f"Could not import model from path '{self.config.model_class_path}'. "
                f"Please check the path and ensure the library is installed."
            ) from e

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Creates and fits the underlying model.

        This follows the scikit-learn convention of instantiating the
        actual model logic within the fit method.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target labels.

        Returns:
            self: The fitted classifier.
        """
        # 1. Instantiate the model here, not in __init__
        self._model = self._create_model()
        
        # 2. Fit the model
        self._model.fit(X, y)
        
        # 3. Expose the 'classes_' attribute after fitting, as required by
        #    scikit-learn for classifiers and used by our evaluator.
        self.classes_ = self._model.classes_
        
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        # Good practice: ensure the model has been fitted before predicting.
        check_is_fitted(self, '_model')
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        """Exposes the feature importances of the underlying model."""
        check_is_fitted(self, '_model')
        if hasattr(self._model, 'feature_importances_'):
            return self._model.feature_importances_
        else:
            raise AttributeError(f"The underlying model {self.config.model_class_path} does not have feature_importances_.")