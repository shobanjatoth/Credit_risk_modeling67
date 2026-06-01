import sys
import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from .logger import get_logger
from .exceptions import CustomException

logger = get_logger(__name__)


class FeatureEngineering:

    def __init__(self, schema):
        try:
            self.schema = schema
            self.target_column = schema["target_column"]
            self.current_year = datetime.now().year
            self.encoder_path = os.path.join("artifacts", "encoder.pkl")

            os.makedirs("artifacts", exist_ok=True)

            logger.info("FeatureEngineering initialized successfully")

        except Exception as e:
            raise CustomException(str(e), sys)

    # --------------------------------------------------
    # Feature Creation
    # --------------------------------------------------
    def create_features(self, df):

        try:
            df = df.copy()

            df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

            # Rename column
            if "Car Name" in df.columns:
                df.rename(columns={"Car Name": "car_name"}, inplace=True)

            # Extract brand & model
            if "car_name" in df.columns:
                df["brand"] = df["car_name"].str.split().str[0]
                df["model"] = df["car_name"].str.split().str[1]

            # Clean fuel
            if "Fuel" in df.columns:
                df["Fuel"] = df["Fuel"].str.capitalize()

            # Extract state
            if "Location" in df.columns:
                df["State"] = df["Location"].str.split("-").str[0]

            df.drop(columns=["car_name", "Location"], inplace=True, errors="ignore")

            # Clean text
            if "brand" in df.columns:
                df["brand"] = df["brand"].str.upper().str.strip()

            if "model" in df.columns:
                df["model"] = df["model"].str.upper().str.strip()

            # Advanced features
            if "Year" in df.columns:
                df["age_of_car"] = self.current_year - df["Year"]

            if "Distance" in df.columns and "age_of_car" in df.columns:
                df["km_per_year"] = df["Distance"] / (df["age_of_car"] + 1)
                df["age_x_distance"] = df["age_of_car"] * df["Distance"]

            # Luxury flag
            if "brand" in df.columns:
                df["is_luxury"] = df["brand"].isin(["BMW", "JEEP"]).astype(int)

            return df

        except Exception as e:
            raise CustomException(str(e), sys)

    # --------------------------------------------------
    # Encoding
    # --------------------------------------------------
    def encode(self, X_train, X_test):

        try:
            categorical_cols = X_train.select_dtypes(include="object").columns

            if len(categorical_cols) == 0:
                return X_train, X_test

            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            )

            encoder.fit(X_train[categorical_cols])

            train_encoded = encoder.transform(X_train[categorical_cols])
            test_encoded = encoder.transform(X_test[categorical_cols])

            train_encoded_df = pd.DataFrame(
                train_encoded,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X_train.index
            )

            test_encoded_df = pd.DataFrame(
                test_encoded,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X_test.index
            )

            X_train = X_train.drop(columns=categorical_cols)
            X_test = X_test.drop(columns=categorical_cols)

            X_train = pd.concat([X_train, train_encoded_df], axis=1)
            X_test = pd.concat([X_test, test_encoded_df], axis=1)

            joblib.dump(encoder, self.encoder_path)

            logger.info("Categorical encoding completed")

            return X_train, X_test

        except Exception as e:
            raise CustomException(str(e), sys)

    # --------------------------------------------------
    # Main Process
    # --------------------------------------------------
    def process(self, train_df, test_df):

        try:
            logger.info("Feature Engineering started")

            train_df = self.create_features(train_df)
            test_df = self.create_features(test_df)

            # Separate target
            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]

            X_train = train_df.drop(columns=[self.target_column])
            X_test = test_df.drop(columns=[self.target_column])

            # Encode
            X_train, X_test = self.encode(X_train, X_test)

            # Save processed data
            X_train.to_csv("artifacts/train_processed.csv", index=False)
            X_test.to_csv("artifacts/test_processed.csv", index=False)

            logger.info("Feature Engineering completed successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(str(e), sys)

