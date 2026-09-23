import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, Optional
from fastapi import APIRouter, FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipe.joblib")


class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class EducationEnum(str, Enum):
    BACHELOR = "Bachelor"
    MASTER = "Master"
    PHD = "PhD"
    HIGH_SCHOOL = "High School"


class CityEnum(str, Enum):
    DHAKA = "Dhaka"
    CHITTAGONG = "Chittagong"
    SYLHET = "Sylhet"
    RAJSHAHI = "Rajshahi"
    KHULNA = "Khulna"


ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        try:
            ml_models["customer_purchase"] = joblib.load(MODEL_PATH)
            print(f"Model 1 loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading joblib model: {e}")
    else:
        print(f"Warning: Model file not found at path: {MODEL_PATH}")

    yield

    ml_models.clear()


app = FastAPI(title="ML Test Service", lifespan=lifespan)

router = APIRouter(prefix="/api/v1")


class CustomerPurchaseInput(BaseModel):
    Age: Optional[float] = Field(
        None, gt=0, description="Customer age in years", json_schema_extra={"example": 32.0}
    )
    Salary: Optional[float] = Field(
        None, ge=0, description="Annual salary", json_schema_extra={"example": 55000.0}
    )
    Experience: Optional[float] = Field(
        None, ge=0, description="Years of work experience", json_schema_extra={"example": 5.0}
    )
    City: Optional[CityEnum] = Field(
        None, description="City name", json_schema_extra={"example": "Dhaka"}
    )
    Gender: Optional[GenderEnum] = Field(
        None, description="Gender", json_schema_extra={"example": "Male"}
    )
    Education: Optional[EducationEnum] = Field(
        None, description="Highest degree attained", json_schema_extra={"example": "Bachelor"}
    )


@router.post("/predict/purchase", tags=["Customer Analytics"])
def predict_purchase(data: CustomerPurchaseInput):
    model = ml_models.get("customer_purchase")
    if not model:
        raise HTTPException(
            status_code=500, detail=f"Customer Purchase model is not loaded. Target path: {MODEL_PATH}"
        )

    try:
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        probabilities = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0].tolist()
            probabilities = {
                str(cls): prob for cls, prob in zip(model.classes_, probs)
            }

        return {
            "model": "customer_purchase",
            "prediction": (
                int(prediction)
                if isinstance(prediction, (int, bool))
                else str(prediction)
            ),
            "probabilities": probabilities,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


app.include_router(router)


@app.get("/", tags=["UI"])
def homepage():
    return {"message": "Welcome! Access interactive API docs at /docs"}