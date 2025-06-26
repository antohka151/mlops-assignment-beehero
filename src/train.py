import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.config.schema import PipelineConfig
from src.data.preprocessor.component import FeaturesPreprocessor
from src.data.outlier_remover import OutlierRemover
from src.models.colony_classifier import ColonyStrengthClassifier
from src.models.pyfunc_wrapper import FullPipelinePyFunc
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig
) -> FullPipelinePyFunc:
    """
    Creates and trains the complete end-to-end model.

    This function acts as a model factory. It takes raw training data and returns
    a single, fitted, deployable model artifact that encapsulates all logic.

    Args:
        X_train: The raw training features.
        y_train: The raw training labels.
        config: The full pipeline configuration object.

    Returns:
        A fitted instance of FullPipelinePyFunc, ready for evaluation and deployment.
    """
    # Step 1: Fit the OutlierRemover and clean the training data
    logger.info("--- Fitting OutlierRemover and cleaning training data ---")
    outlier_remover = OutlierRemover(config.outlier_remover)
    outlier_remover.fit(X_train)
    X_train_clean, y_train_clean = outlier_remover.transform(X_train, y_train)
    logger.info(f"Data shape after outlier removal: {X_train_clean.shape}")

    # Step 2: Define and train the core scikit-learn pipeline
    logger.info("--- Assembling and Training the core scikit-learn pipeline ---")
    sklearn_pipeline = Pipeline(steps=[
        ('preprocessor', FeaturesPreprocessor(steps=config.data_preprocessor)),
        ('feature_selector', ColumnTransformer(
            transformers=[('selector', 'passthrough', config.training.feature_columns)],
            remainder='drop',
            verbose_feature_names_out=False
        )),
        ('classifier', ColonyStrengthClassifier(config.model))
    ])
    sklearn_pipeline.fit(X_train_clean, y_train_clean)

    # Step 3: Assemble the final PyFunc model with the fitted components
    logger.info("--- Assembling the final FullPipelinePyFunc model ---")
    final_model = FullPipelinePyFunc(
        outlier_remover=outlier_remover,
        sklearn_pipeline=sklearn_pipeline
    )

    logger.info("--- Model training and assembly complete ---")
    return final_model