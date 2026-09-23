from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from fastapi import APIRouter, FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum



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
    try:
        model_path = "api/models/pipe.joblib"
        ml_models["customer_purchase"] = joblib.load(model_path)
        print("Model 1 loaded")
    except FileNotFoundError:
        print("Warning: model 1 not found in directory.")

    yield

    ml_models.clear()


app = FastAPI(title="ML Test Service", lifespan=lifespan)

router = APIRouter(prefix="/api/v1")


class CustomerPurchaseInput(BaseModel):
    Age: Optional[float] = Field(
        None, gt=0, description="Customer age in years", example=32.0
    )
    Salary: Optional[float] = Field(
        None, ge=0, description="Annual salary", example=55000.0
    )
    Experience: Optional[float] = Field(
        None, ge=0, description="Years of work experience", example=5.0
    )
    City: Optional[str] = Field(
        None, description="City name", example="Dhaka"
    )
    Gender: Optional[str] = Field(
        None, description="Gender", example="Male"
    )
    Education: Optional[str] = Field(
        None, description="Highest degree attained", example="Bachelor"
    )



@router.post("/predict/purchase", tags=["Customer Analytics"])
def predict_purchase(data: CustomerPurchaseInput):
    model = ml_models.get("customer_purchase")
    if not model:
        raise HTTPException(
            status_code=500, detail="Customer Purchase model is not loaded."
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])

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