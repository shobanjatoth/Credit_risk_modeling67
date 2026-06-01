import pandas as pd
from src.pipelines.inference_pipeline import InferencePipeline

if __name__ == "__main__":

    sample = {
    "Car Name": "Hyundai Creta",
    "Year": 2018,
    "Distance": 40000,
    "Owner": 1,
    "Fuel": "Petrol",
    "Location": "Hyderabad-Telangana",
    "Drive": "Manual",
    "Type": "SUV"
    }


    df = pd.DataFrame([sample])

    pipeline = InferencePipeline("config/config.yaml")

    prediction = pipeline.predict(df)

    print("Prediction:", prediction)

