import sys
import joblib
import pandas as pd
import numpy as np

from src.utils.utils import load_yaml
from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException
from src.components.feature_engineering import FeatureEngineering

logger = get_logger(__name__)


class InferencePipeline:

    def __init__(self, config_path: str):
        try:
            self.config = load_yaml(config_path)
            self.schema = load_yaml("config/schema.yaml")

            self.model_path = self.config["artifacts"]["model_path"]
            self.encoder_path = self.config["artifacts"]["encoder_path"]
            self.selected_features_path = (
                "artifacts/selected_features.pkl"
            )

            # Load artifacts
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)
            self.selected_features = joblib.load(
                self.selected_features_path
            )

            # Feature Engineering
            self.fe = FeatureEngineering(
                self.config,
                self.schema
            )

            # Threshold tuning
            self.threshold_cfg = self.config.get(
                "threshold_tuning",
                {}
            )

            logger.info(
                "Inference pipeline initialized successfully"
            )

        except Exception as e:
            raise CustomException(str(e), sys)

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------
    def preprocess(self, input_df: pd.DataFrame):

        try:
            logger.info(
                "Starting preprocessing for inference"
            )

            df = input_df.copy()

            # Remove unwanted columns
            df.drop(
                columns=[
                    "Unnamed: 0",
                    "PROSPECTID",
                    "Credit_Score"
                ],
                errors="ignore",
                inplace=True
            )

            # Apply cleaning
            df = self.fe.clean(df)

            # -----------------------------------
            # Encode categorical columns
            # -----------------------------------
            categorical_cols = list(
                self.encoder.feature_names_in_
            )

            for col in categorical_cols:
                if col not in df.columns:
                    df[col] = "Unknown"

            encoded = self.encoder.transform(
                df[categorical_cols]
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(
                    categorical_cols
                ),
                index=df.index
            )

            # Remove original categorical columns
            df.drop(
                columns=categorical_cols,
                inplace=True,
                errors="ignore"
            )

            # Merge encoded features
            df = pd.concat(
                [df, encoded_df],
                axis=1
            )

            # -----------------------------------
            # Numeric safety
            # -----------------------------------
            df = df.apply(
                pd.to_numeric,
                errors="coerce"
            ).fillna(0)

            # -----------------------------------
            # Match exact model features
            # -----------------------------------
            model_features = (
                self.model
                .get_booster()
                .feature_names
            )

            for col in model_features:
                if col not in df.columns:
                    df[col] = 0

            df = df.reindex(
                columns=model_features,
                fill_value=0
            )

            logger.info(
                f"Preprocessing completed. Shape: {df.shape}"
            )

            return df

        except Exception as e:
            raise CustomException(str(e), sys)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    def predict(self, input_df: pd.DataFrame):

        try:
            processed_df = self.preprocess(
                input_df
            )

            probs = self.model.predict_proba(
                processed_df
            )

            preds = np.argmax(
                probs,
                axis=1
            )

            # -----------------------------------
            # Threshold tuning
            # -----------------------------------
            if self.threshold_cfg.get(
                "enable",
                False
            ):

                target_class = self.threshold_cfg.get(
                    "target_class",
                    2
                )

                threshold = self.threshold_cfg.get(
                    "default_threshold",
                    0.25
                )

                for i in range(len(preds)):

                    if (
                        probs[i][target_class]
                        >= threshold
                    ):
                        preds[i] = target_class

            logger.info(
                "Prediction completed successfully"
            )

            return (
                preds.tolist(),
                probs.tolist()
            )

        except Exception as e:
            raise CustomException(str(e), sys)