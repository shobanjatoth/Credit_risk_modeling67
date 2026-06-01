# import sys
# import os
# import json
# import joblib

# from sklearn.metrics import classification_report, f1_score, accuracy_score

# from src.logger.logger import get_logger
# from src.exceptions.exceptions import CustomException

# logger = get_logger(__name__)


# class ModelEvaluation:

#     def __init__(self, config):
#         self.metrics_path = config["artifacts"]["metrics_path"]
#         self.feature_path = "artifacts/selected_features.pkl"

#         os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

#     # ---------------- EVALUATION ----------------
#     def evaluate(self, model, X_test, y_test):
#         try:
#             logger.info("Model Evaluation Started")

#             # 🔥 LOAD SAME FEATURES USED IN TRAINING
#             selected_features = joblib.load(self.feature_path)

#             # 🔥 ALIGN FEATURES (CRITICAL FIX)
#             X_test = X_test[selected_features]

#             # Predict
#             preds = model.predict(X_test)

#             # Metrics
#             metrics = {
#                 "accuracy": float(accuracy_score(y_test, preds)),
#                 "f1_score": float(f1_score(y_test, preds, average="weighted"))
#             }

#             # Save metrics
#             with open(self.metrics_path, "w") as f:
#                 json.dump(metrics, f, indent=4)

#             print("\n📊 Classification Report:\n")
#             print(classification_report(y_test, preds))

#             logger.info("Model Evaluation Completed")

#             return metrics

#         except Exception as e:
#             raise CustomException(str(e), sys)


import sys
import os
import json

from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score
)

from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


class ModelEvaluation:

    def __init__(self, config):

        self.metrics_path = config["artifacts"]["metrics_path"]

        os.makedirs(
            os.path.dirname(self.metrics_path),
            exist_ok=True
        )

    # =====================================================
    # EVALUATION
    # =====================================================
    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            logger.info("Model evaluation started")

            # -----------------------------------------
            # PREDICTIONS
            # -----------------------------------------
            preds = model.predict(X_test)

            # -----------------------------------------
            # METRICS
            # -----------------------------------------
            accuracy = accuracy_score(
                y_test,
                preds
            )

            weighted_f1 = f1_score(
                y_test,
                preds,
                average="weighted"
            )

            report = classification_report(
                y_test,
                preds,
                output_dict=True
            )

            metrics = {
                "accuracy": float(accuracy),
                "weighted_f1": float(weighted_f1),
                "classification_report": report
            }

            # -----------------------------------------
            # SAVE METRICS
            # -----------------------------------------
            with open(self.metrics_path, "w") as f:

                json.dump(
                    metrics,
                    f,
                    indent=4
                )

            print("\n" + "=" * 60)
            print("CLASSIFICATION REPORT")
            print("=" * 60)

            print(
                classification_report(
                    y_test,
                    preds
                )
            )

            logger.info("Model evaluation completed")

            return metrics

        except Exception as e:
            raise CustomException(str(e), sys)