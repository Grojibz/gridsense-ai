"""`/predict` route — DegradeML battery State-of-Health prediction."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gridsense.degrade.serve import predict_soh

router = APIRouter(tags=["degrade"])


class PredictRequest(BaseModel):
    cycle_count: float = Field(ge=0, description="Equivalent charge/discharge cycles.")
    avg_temperature_c: float = Field(description="Average operating temperature (°C).")
    avg_dod: float = Field(ge=0, le=1, description="Average depth of discharge (0–1).")
    avg_c_rate: float = Field(ge=0, description="Average C-rate.")
    calendar_age_days: float = Field(ge=0, description="Calendar age in days.")


class PredictResponse(BaseModel):
    predicted_soh: float = Field(description="Predicted State of Health (%).")
    model_version: str = Field(description="Registered model version used.")


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict battery State of Health from operating conditions."""
    result = predict_soh(request.model_dump())
    return PredictResponse(**result)
