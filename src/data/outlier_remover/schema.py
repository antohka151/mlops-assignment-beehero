from pydantic import BaseModel, Field

class OutlierRemoverConfig(BaseModel):
    """Configuration for the outlier removal step."""
    group_by_column: str = Field(..., description="Column to group by for temperature calculations.")
    temperature_column: str = Field(..., description="Column with temperature readings.")
    max_temp_threshold: float = Field(..., description="Upper threshold for mean temperature to filter outliers.")
    min_temp_threshold: float = Field(..., description="Lower threshold for mean temperature to filter outliers.")