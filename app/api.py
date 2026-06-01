from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.pipelines.inference_pipeline import InferencePipeline

app = FastAPI(title="Credit Risk Modeling API")

# Load model pipeline
pipeline = InferencePipeline(
    config_path="config/config.yaml"
)


class CreditRiskInput(BaseModel):
    Age_Oldest_TL: float
    enq_L3m: float
    time_since_recent_enq: float
    enq_L6m: float
    num_std: float
    num_std_12mts: float
    Time_With_Curr_Empr: float
    time_since_recent_payment: float
    AGE: float
    enq_L12m: float
    NETMONTHLYINCOME: float
    Age_Newest_TL: float
    pct_currentBal_all_TL: float
    num_std_6mts: float
    time_since_recent_deliquency: float
    Total_TL: float
    tot_enq: float
    max_unsec_exposure_inPct: float
    time_since_first_deliquency: float
    pct_PL_enq_L6m_of_L12m: float
    max_delinquency_level: float
    Secured_TL: float
    Tot_Closed_TL: float
    pct_tl_open_L12M: float
    pct_PL_enq_L6m_of_ever: float
    max_recent_level_of_deliq: float
    PL_enq_L6m: float
    recent_level_of_deliq: float
    Other_TL: float
    pct_active_tl: float


@app.get("/")
def home():
    return {
        "message": "Credit Risk Modeling API is running"
    }


@app.post("/predict")
def predict(data: CreditRiskInput):

    try:
        # Convert request to DataFrame
        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        # Predict
        prediction, probabilities = pipeline.predict(
            input_df
        )

        return {
            "prediction": int(prediction[0]),
            "probabilities": probabilities[0]
        }

    except Exception as e:
        return {
            "error": str(e)
        }