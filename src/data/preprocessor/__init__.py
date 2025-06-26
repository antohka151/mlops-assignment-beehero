"""
Feature Preprocessor Package.

This package provides a configurable pipeline for feature engineering.

Exposes:
- FeaturesPreprocessor: The main pipeline orchestrator.
- PreprocessorStepConfig: The Pydantic model for configuring a pipeline step.
- StepMetadataConfig: The Pydantic model for metadata about executed steps.
"""
from .component import FeaturesPreprocessor
from .schema import PreprocessorStepConfig, StepMetadataConfig

__all__ = [
    "FeaturesPreprocessor",
    "PreprocessorStepConfig",
    "StepMetadataConfig",
]