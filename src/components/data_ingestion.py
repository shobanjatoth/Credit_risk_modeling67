import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


class DataIngestion:

    def __init__(self, config):
        self.config = config
        self.raw_path = config["data"]["raw_data_path"]
        self.test_size = config["data"]["test_size"]
        self.random_state = config["training"]["random_state"]

        self.train_path = config["artifacts"]["train_data_path"]
        self.test_path = config["artifacts"]["test_data_path"]

        os.makedirs("artifacts", exist_ok=True)

    def load_data(self):
        try:
            if not os.path.exists(self.raw_path):
                raise FileNotFoundError(f"{self.raw_path} not found")

            df = pd.read_csv(self.raw_path)
            logger.info("Raw data loaded successfully")
            return df

        except Exception as e:
            raise CustomException(str(e), sys)

    def split_data(self, df, target_column):
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state
            )

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)

            logger.info("Data split completed")

            return train_df, test_df

        except Exception as e:
            raise CustomException(str(e), sys)
