import sys
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .logger import get_logger
from .exceptions import CustomException

logger = get_logger(__name__)


class ModelEvaluation:

    def __init__(self, config):
        self.config = config
        self.metrics_path = config["evaluation"]["metrics_path"]

    def evaluate(self, model, X_test, y_test):

        try:
            predictions = model.predict(X_test)

            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            metrics = {
                "r2_score": r2,
                "mae": mae,
                "mse": mse
            }

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info("Model evaluation completed")

            return metrics

        except Exception as e:
            logger.error(e)
            raise CustomException(str(e), sys)
