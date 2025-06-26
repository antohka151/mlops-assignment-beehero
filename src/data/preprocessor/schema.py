"""
Defines the data contracts for the feature preprocessing pipeline.

This includes the configuration models for defining pipeline steps and
the metadata models for describing the results of an executed pipeline.
"""
from pydantic import BaseModel, Field
from typing import Any

class PreprocessorStepConfig(BaseModel):
    """
    Defines the configuration for a single step in the feature pipeline.
    This is the INPUT schema used to build the FeaturesPreprocessor.
    """
    name: str = Field(..., description="A unique, human-readable name for the step.")
    class_path: str = Field(
        ..., 
        description="The import path to the transformer class. Can be a short form "
                    "(e.g., 'feature_store.ClassName') or a full path."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters to be passed to the transformer's constructor."
    )

class StepMetadataConfig(BaseModel):
    """
    Defines the structure for the metadata of a single executed step.
    This is the OUTPUT schema from the get_metadata() method.
    """
    name: str = Field(description="The unique name of the step from the config.")
    class_path: str = Field(description="The import path used to load the transformer class.")
    params: dict[str, Any] = Field(description="The parameters used to initialize the transformer.")
    source_hash: str = Field(description="A SHA256 hash of the transformer class's source code for versioning.")
