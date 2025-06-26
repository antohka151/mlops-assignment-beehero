import hashlib
import inspect
import importlib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger
from .feature_store import FeatureTransformer
from .schema import PreprocessorStepConfig, StepMetadataConfig

logger = get_logger(__name__)

def _create_transformer_from_config(config: PreprocessorStepConfig) -> FeatureTransformer:
    """
    Resolves the class path, imports, and instantiates a transformer from its config.
    """
    DEFAULT_MODULE_PREFIX = "src.data.preprocessor." # Adjust if needed
    class_path = config.class_path
    if '.' in class_path and not class_path.startswith(DEFAULT_MODULE_PREFIX):
        module_name = class_path.split('.')[0]
        if module_name != "src":
            class_path = f"{DEFAULT_MODULE_PREFIX}{class_path}"
            logger.debug(f"Expanded short path '{config.class_path}' to '{class_path}'")

    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        transformer_class = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import class '{config.class_path}'.") from e

    return transformer_class(**config.params)


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    """
    Orchestrates a sequence of feature transformations based on a validated
    pipeline configuration. This class is a scikit-learn compatible transformer.
    """
    def __init__(self, steps: list[PreprocessorStepConfig]):
        self.steps = steps
        logger.info("Initialized FeaturesPreprocessor with steps: %s", [step.name for step in steps])

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        """
        Fits all transformers sequentially. Each step is fitted on the
        output of the previous step's transformation. The 'y' parameter is
        ignored but included for compatibility with scikit-learn.
        """
        logger.info("Fitting FeaturesPreprocessor...")
        self.fitted_steps_: list[FeaturesPreprocessor] = []
        
        df_temp = X.copy()

        for step_config in self.steps:
            step_name = step_config.name
            logger.info(f"  - Fitting step: '{step_name}'")
            
            transformer = _create_transformer_from_config(step_config)

            transformer.fit(df_temp)
            self.fitted_steps_.append(transformer)
            
            df_temp = transformer.transform(df_temp)

        logger.info("FeaturesPreprocessor fitting complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using the already fitted pipeline of transformers.
        """
        if not hasattr(self, 'fitted_steps_'):
            raise RuntimeError("This preprocessor instance is not fitted yet. Call 'fit' before 'transform'.")
        
        logger.info("Transforming data using fitted FeaturesPreprocessor...")
        df_current = X.copy()

        for step in self.fitted_steps_:
            step_name = step.__class__.__name__
            logger.info(f"  - Applying step: '{step_name}'")
            df_current = step.transform(df_current)
                
        return df_current

    def get_metadata(self) -> list[StepMetadataConfig]:
        """Returns structured metadata about each fitted step for external logging."""
        if not hasattr(self, 'fitted_steps_') or not self.fitted_steps_:
            raise RuntimeError("Pipeline has not been fitted. Call 'get_metadata' after 'fit' or 'fit_transform'.")

        metadata_list = []
        for config, step_instance in zip(self.steps, self.fitted_steps_):
            source_code = inspect.getsource(step_instance.__class__)
            class_hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
            
            metadata = StepMetadataConfig(
                name=config.name,
                class_path=config.class_path,
                params=config.params,
                source_hash=class_hash
            )
            metadata_list.append(metadata)
        return metadata_list