import sys
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


class FeatureEngineering:

    def __init__(self, config, schema):

        self.config = config
        self.schema = schema

        self.target_column = schema["target_column"]

        # =========================
        # Paths
        # =========================
        self.encoder_path = config["artifacts"]["encoder_path"]

        self.train_processed_path = config["artifacts"]["train_processed_path"]
        self.test_processed_path = config["artifacts"]["test_processed_path"]

        self.selected_features_path = "artifacts/selected_features.pkl"
        self.label_encoder_path = "artifacts/label_encoder.pkl"

        os.makedirs("artifacts", exist_ok=True)

    # =========================================================
    # CLEANING
    # =========================================================
    def clean(self, df):

        try:
            df = df.copy()

            logger.info("Starting data cleaning")

            # ---------------------------------
            # Drop unwanted columns
            # ---------------------------------
            drop_cols = [
                "Unnamed: 0",
                "PROSPECTID",
                "Credit_Score"
            ]

            df.drop(
                columns=drop_cols,
                inplace=True,
                errors="ignore"
            )

            # ---------------------------------
            # Remove invalid values
            # ---------------------------------
            if "Age_Oldest_TL" in df.columns:
                df = df[df["Age_Oldest_TL"] != -99999]

            # ---------------------------------
            # Remove duplicates
            # ---------------------------------
            df.drop_duplicates(inplace=True)

            # ---------------------------------
            # EDUCATION ordinal mapping
            # ---------------------------------
            education_map = {
                "SSC": 1,
                "12TH": 2,
                "GRADUATE": 3,
                "UNDER GRADUATE": 3,
                "POST-GRADUATE": 4,
                "OTHERS": 1,
                "PROFESSIONAL": 3
            }

            if "EDUCATION" in df.columns:

                df["EDUCATION"] = (
                    df["EDUCATION"]
                    .map(education_map)
                    .fillna(0)
                    .astype(int)
                )

            logger.info("Data cleaning completed")

            return df

        except Exception as e:
            raise CustomException(str(e), sys)

    # =========================================================
    # ENCODING
    # =========================================================
    def encode(self, X_train, X_test, y_train, y_test):

        try:

            logger.info("Starting encoding process")

            # ---------------------------------
            # Categorical columns
            # ---------------------------------
            categorical_cols = [
                "MARITALSTATUS",
                "GENDER",
                "last_prod_enq2",
                "first_prod_enq2"
            ]

            categorical_cols = [
                col for col in categorical_cols
                if col in X_train.columns
            ]

            # ---------------------------------
            # One Hot Encoder
            # ---------------------------------
            encoder = OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore"
            )

            encoder.fit(X_train[categorical_cols])

            train_encoded = encoder.transform(
                X_train[categorical_cols]
            )

            test_encoded = encoder.transform(
                X_test[categorical_cols]
            )

            encoded_columns = encoder.get_feature_names_out(
                categorical_cols
            )

            train_encoded_df = pd.DataFrame(
                train_encoded,
                columns=encoded_columns,
                index=X_train.index
            )

            test_encoded_df = pd.DataFrame(
                test_encoded,
                columns=encoded_columns,
                index=X_test.index
            )

            # ---------------------------------
            # Drop original categorical columns
            # ---------------------------------
            X_train.drop(
                columns=categorical_cols,
                inplace=True
            )

            X_test.drop(
                columns=categorical_cols,
                inplace=True
            )

            # ---------------------------------
            # Concatenate encoded columns
            # ---------------------------------
            X_train = pd.concat(
                [X_train, train_encoded_df],
                axis=1
            )

            X_test = pd.concat(
                [X_test, test_encoded_df],
                axis=1
            )

            # ---------------------------------
            # Label Encoding target
            # ---------------------------------
            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            # ---------------------------------
            # Save encoders
            # ---------------------------------
            joblib.dump(
                encoder,
                self.encoder_path
            )

            joblib.dump(
                label_encoder,
                self.label_encoder_path
            )

            logger.info("Encoding completed successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(str(e), sys)

    # =========================================================
    # FEATURE SELECTION
    # =========================================================
    def feature_selection(
        self,
        X_train,
        y_train,
        X_test
    ):

        try:

            logger.info("Starting feature selection")

            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)

            feature_importance_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": rf.feature_importances_
            })

            feature_importance_df = feature_importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            # ---------------------------------
            # Select Top Features
            # ---------------------------------
            top_k = 30

            selected_features = (
                feature_importance_df
                .head(top_k)["Feature"]
                .tolist()
            )

            logger.info(
                f"Top {top_k} features selected"
            )

            # ---------------------------------
            # Save selected feature names
            # ---------------------------------
            joblib.dump(
                selected_features,
                self.selected_features_path
            )

            # ---------------------------------
            # Filter train/test
            # ---------------------------------
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

            logger.info("Feature selection completed")

            return X_train, X_test

        except Exception as e:
            raise CustomException(str(e), sys)

    # =========================================================
    # FULL PROCESSING PIPELINE
    # =========================================================
    def process(self, train_df, test_df):

        try:

            logger.info("Starting feature engineering pipeline")

            # ---------------------------------
            # Cleaning
            # ---------------------------------
            train_df = self.clean(train_df)
            test_df = self.clean(test_df)

            # ---------------------------------
            # Split features and target
            # ---------------------------------
            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]

            X_train = train_df.drop(
                columns=[self.target_column]
            )

            X_test = test_df.drop(
                columns=[self.target_column]
            )

            # ---------------------------------
            # Encoding
            # ---------------------------------
            X_train, X_test, y_train, y_test = self.encode(
                X_train,
                X_test,
                y_train,
                y_test
            )

            # ---------------------------------
            # Feature Selection
            # ---------------------------------
            X_train, X_test = self.feature_selection(
                X_train,
                y_train,
                X_test
            )

            # ---------------------------------
            # Save processed files
            # ---------------------------------
            train_processed = pd.concat(
                [
                    X_train,
                    pd.Series(
                        y_train,
                        name=self.target_column
                    )
                ],
                axis=1
            )

            test_processed = pd.concat(
                [
                    X_test,
                    pd.Series(
                        y_test,
                        name=self.target_column
                    )
                ],
                axis=1
            )

            train_processed.to_csv(
                self.train_processed_path,
                index=False
            )

            test_processed.to_csv(
                self.test_processed_path,
                index=False
            )

            logger.info(
                "Feature engineering pipeline completed"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(str(e), sys)