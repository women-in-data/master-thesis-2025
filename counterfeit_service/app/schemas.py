from pydantic import BaseModel, Field
from typing import Optional


class Signals(BaseModel):
    multimodal_score: float = Field(..., description="Full fusion model probability")
    image_signal: float = Field(..., description="Image-only contribution (img features, others zeroed)")
    text_signal: float = Field(..., description="Text-only contribution (d2v features, others zeroed)")


class PredictionResponse(BaseModel):
    is_counterfeit: bool
    probability: float
    signals: Signals
