# import sys
# import mlflow
# import mlflow.sklearn

# from src.utils.utils import load_yaml
# from src.components.data_ingestion import DataIngestion
# from src.components.data_validation import DataValidation
# from src.components.feature_engineering import FeatureEngineering
# from src.components.model_trainer import ModelTrainer
# from src.components.model_evaluation import ModelEvaluation

# from src.logger.logger import get_logger
# from src.exceptions.exceptions import CustomException

# logger = get_logger(__name__)


# class TrainingPipeline:

#     def __init__(self, config_path: str, schema_path: str):
#         try:
#             self.config = load_yaml(config_path)
#             self.schema = load_yaml(schema_path)

#             # ✅ MLflow config extraction
#             self.experiment_name = self.config["mlflow"]["experiment_name"]
#             self.registered_model_name = self.config["mlflow"]["registered_model_name"]

#         except Exception as e:
#             raise CustomException(str(e), sys)

#     # --------------------------------------------------
#     # MAIN PIPELINE
#     # --------------------------------------------------
#     def run_pipeline(self):
#         try:
#             logger.info("Training Pipeline started")

#             # -------------------------------
#             # Data Ingestion
#             # -------------------------------
#             ingestion = DataIngestion(self.config)
#             df = ingestion.load_data()

#             # -------------------------------
#             # Data Validation
#             # -------------------------------
#             validator = DataValidation(
#                 self.schema,
#                 report_path="artifacts/validation_report.json"
#             )

#             status = validator.validate(df)

#             if not status:
#                 raise Exception("Data validation failed. Check report.")

#             # -------------------------------
#             # Train/Test Split
#             # -------------------------------
#             train_df, test_df = ingestion.split_data(
#                 df,
#                 target_column=self.config["target_column"]
#             )

#             # -------------------------------
#             # Feature Engineering
#             # -------------------------------
#             fe = FeatureEngineering(self.config, self.schema)

#             X_train, X_test, y_train, y_test = fe.process(
#                 train_df, test_df
#             )

#             # -------------------------------
#             # MLflow Setup
#             # -------------------------------
#             mlflow.set_experiment(self.experiment_name)

#             with mlflow.start_run():

#                 # -------------------------------
#                 # Model Training
#                 # -------------------------------
#                 trainer = ModelTrainer(self.config)

#                 model = trainer.run(
#                     X_train,
#                     X_test,
#                     y_train,
#                     y_test
#                 )

#                 # -------------------------------
#                 # Model Evaluation
#                 # -------------------------------
#                 evaluator = ModelEvaluation(self.config)

#                 metrics = evaluator.evaluate(
#                     model,
#                     X_test,
#                     y_test
#                 )

#                 # -------------------------------
#                 # Log parameters
#                 # -------------------------------
#                 mlflow.log_params(self.config["training"])

#                 # -------------------------------
#                 # Log metrics
#                 # -------------------------------
#                 mlflow.log_metrics(metrics)

#                 # -------------------------------
#                 # Log & Register Model
#                 # -------------------------------
#                 mlflow.sklearn.log_model(
#                     sk_model=model,
#                     artifact_path="model",
#                     registered_model_name=self.registered_model_name
#                 )

#             logger.info("Training Pipeline completed successfully")

#         except Exception as e:
#             logger.error(f"Pipeline failed: {e}")
#             raise CustomException(str(e), sys)


import sys
import os

import mlflow
import mlflow.sklearn

from src.utils.utils import load_yaml

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


class TrainingPipeline:

    def __init__(
        self,
        config_path: str,
        schema_path: str
    ):

        try:

            # =================================================
            # LOAD CONFIG FILES
            # =================================================
            self.config = load_yaml(config_path)

            self.schema = load_yaml(schema_path)

            # =================================================
            # MLFLOW CONFIG
            # =================================================
            self.experiment_name = (
                self.config["mlflow"]["experiment_name"]
            )

            self.registered_model_name = (
                self.config["mlflow"]["registered_model_name"]
            )

        except Exception as e:
            raise CustomException(str(e), sys)

    # =====================================================
    # MAIN TRAINING PIPELINE
    # =====================================================
    def run_pipeline(self):

        try:

            logger.info("Training pipeline started")

            # =================================================
            # DATA INGESTION
            # =================================================
            ingestion = DataIngestion(
                self.config
            )

            df = ingestion.load_data()

            # =================================================
            # DATA VALIDATION
            # =================================================
            validator = DataValidation(
                self.schema,
                report_path="artifacts/validation_report.json"
            )

            validation_status = validator.validate(df)

            if not validation_status:

                raise CustomException(
                    "Data validation failed. Check validation report.",
                    sys
                )

            logger.info("Data validation completed")

            # =================================================
            # TRAIN TEST SPLIT
            # =================================================
            train_df, test_df = ingestion.split_data(
                df,
                target_column=self.config["target_column"]
            )

            logger.info("Train test split completed")

            # =================================================
            # FEATURE ENGINEERING
            # =================================================
            fe = FeatureEngineering(
                self.config,
                self.schema
            )

            X_train, X_test, y_train, y_test = fe.process(
                train_df,
                test_df
            )

            logger.info("Feature engineering completed")

            # =================================================
            # SET MLFLOW EXPERIMENT
            # =================================================
            mlflow.set_experiment(
                self.experiment_name
            )

            with mlflow.start_run():

                logger.info("MLflow run started")

                # =============================================
                # MODEL TRAINING
                # =============================================
                trainer = ModelTrainer(
                    self.config
                )

                model = trainer.run(
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )

                logger.info("Model training completed")

                # =============================================
                # MODEL EVALUATION
                # =============================================
                evaluator = ModelEvaluation(
                    self.config
                )

                metrics = evaluator.evaluate(
                    model,
                    X_test,
                    y_test
                )

                logger.info("Model evaluation completed")

                # =============================================
                # LOG PARAMETERS
                # =============================================
                mlflow.log_params(
                    self.config["training"]
                )

                # =============================================
                # LOG METRICS
                # =============================================
                mlflow.log_metric(
                    "accuracy",
                    metrics["accuracy"]
                )

                mlflow.log_metric(
                    "weighted_f1",
                    metrics["weighted_f1"]
                )

                logger.info("Metrics logged to MLflow")

                # =============================================
                # LOG ARTIFACTS SAFELY
                # =============================================
                artifact_files = [

                    "artifacts/metrics.json",

                    "artifacts/validation_report.json",

                    "artifacts/feature_importance.csv"
                ]

                for file in artifact_files:

                    if os.path.exists(file):

                        mlflow.log_artifact(file)

                        logger.info(
                            f"Artifact logged: {file}"
                        )

                    else:

                        logger.warning(
                            f"Artifact not found: {file}"
                        )

                # =============================================
                # LOG & REGISTER MODEL
                # =============================================
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.registered_model_name
                )

                logger.info("Model logged to MLflow")

            logger.info(
                "Training pipeline completed successfully"
            )

        except Exception as e:

            logger.error(
                f"Training pipeline failed: {e}"
            )

            raise CustomException(str(e), sys)