import sys
import mlflow
from .utils import load_yaml
from .data_ingestion import DataIngestion
from .data_validation import DataValidation
from .feature_engineering import FeatureEngineering
from .model_trainer import ModelTrainer
from .model_evaluation import ModelEvaluation
from .logger import get_logger
from .exceptions import CustomException

logger = get_logger(__name__)


class TrainingPipeline:

    def __init__(self):
        self.config = load_yaml("config/config.yaml")
        self.schema = load_yaml("config/schema.yaml")

    def run(self):

        try:
            logger.info("Training pipeline started")

            # 1️⃣ Ingestion
            ingestion = DataIngestion(self.config)
            df = ingestion.load_data()

            # 2️⃣ Validation
            validator = DataValidation(
                self.schema,
                "artifacts/validation_report.json"
            )

            if not validator.validate(df):
                raise Exception("Data validation failed")

            # 3️⃣ Split
            train_df, test_df = ingestion.split_data(
                df,
                self.schema["target_column"]
            )

            # 4️⃣ Feature Engineering
            fe = FeatureEngineering(self.schema)

            X_train, X_test, y_train, y_test = fe.process(
                train_df,
                test_df
            )

            # 5️⃣ MLflow
            mlflow.set_experiment(
                self.config["mlflow"]["experiment_name"]
            )

            with mlflow.start_run():

                # 6️⃣ Train
                trainer = ModelTrainer(self.config)
                model = trainer.train(X_train, y_train)

                mlflow.log_params(self.config["training"])

                # 7️⃣ Evaluate
                evaluator = ModelEvaluation(self.config)
                metrics = evaluator.evaluate(model, X_test, y_test)

                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=self.config["mlflow"]["registered_model_name"]
                )

            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error(e)
            raise CustomException(str(e), sys)

