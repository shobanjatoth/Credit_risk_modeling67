import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from .logger import get_logger
from .exceptions import CustomException

logger = get_logger(__name__)


class ModelTrainer:

    def __init__(self, config):
        self.config = config
        self.model_path = config["model"]["model_path"]
        self.n_estimators = config["training"]["n_estimators"]
        self.max_depth = config["training"]["max_depth"]

    def train(self, X_train, y_train):

        try:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.config["training"]["random_state"]
            )

            model.fit(X_train, y_train)

            joblib.dump(model, self.model_path)
            logger.info("Model training completed")

            return model

        except Exception as e:
            logger.error(e)
            raise CustomException(str(e), sys)
