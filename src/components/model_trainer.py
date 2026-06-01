# import sys
# import os
# import pandas as pd
# import joblib

# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from imblearn.combine import SMOTETomek

# from src.logger.logger import get_logger
# from src.exceptions.exceptions import CustomException

# logger = get_logger(__name__)


# class ModelTrainer:

#     def __init__(self, config):
#         self.config = config

#         self.model_path = config["artifacts"]["model_path"]
#         self.feature_path = "artifacts/selected_features.pkl"

#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

#     # ---------------- FEATURE SELECTION ----------------
#     def select_features(self, X_train, y_train, X_test):

#         rf = RandomForestClassifier(random_state=42)
#         rf.fit(X_train, y_train)

#         feat_df = pd.DataFrame({
#             "Feature": X_train.columns,
#             "Importance": rf.feature_importances_
#         }).sort_values(by="Importance", ascending=False)

#         top_features = feat_df.head(30)["Feature"].tolist()

#         # ✅ SAVE FEATURES (IMPORTANT FIX)
#         joblib.dump(top_features, self.feature_path)

#         return X_train[top_features], X_test[top_features]

#     # ---------------- SMOTE ----------------
#     def apply_smote(self, X_train, y_train):
#         sm = SMOTETomek(random_state=42)
#         return sm.fit_resample(X_train, y_train)

#     # ---------------- TRAIN MODEL ----------------
#     def train_model(self, X_train, y_train):

#         model = XGBClassifier(
#             random_state=self.config["training"]["random_state"],
#             n_estimators=self.config["training"]["n_estimators"],
#             max_depth=self.config["training"]["max_depth"],
#             learning_rate=self.config["training"]["learning_rate"],
#             subsample=self.config["training"]["subsample"],
#             colsample_bytree=self.config["training"]["colsample_bytree"],
#             gamma=self.config["training"]["gamma"],
#             reg_alpha=self.config["training"]["reg_alpha"],
#             reg_lambda=self.config["training"]["reg_lambda"],
#             min_child_weight=self.config["training"]["min_child_weight"],
#             objective=self.config["training"]["objective"],
#             eval_metric=self.config["training"]["eval_metric"]
#         )

#         model.fit(X_train, y_train)
#         return model

#     # ---------------- PIPELINE ----------------
#     def run(self, X_train, X_test, y_train, y_test):

#         logger.info("Model Training Started")

#         # Step 1: feature selection
#         X_train, X_test = self.select_features(X_train, y_train, X_test)

#         # Step 2: SMOTE
#         X_train, y_train = self.apply_smote(X_train, y_train)

#         # Step 3: train model
#         model = self.train_model(X_train, y_train)

#         # Step 4: save model
#         joblib.dump(model, self.model_path)

#         logger.info("Model Training Completed")

#         return model



import sys
import os

from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


class ModelTrainer:

    def __init__(self, config):

        self.config = config

    # =====================================================
    # APPLY SMOTE
    # =====================================================
    def apply_smote(self, X_train, y_train):

        try:
            logger.info("Applying SMOTE + Tomek")

            smote = SMOTETomek(
                random_state=self.config["training"]["random_state"]
            )

            X_resampled, y_resampled = smote.fit_resample(
                X_train,
                y_train
            )

            logger.info("SMOTE completed")

            return X_resampled, y_resampled

        except Exception as e:
            raise CustomException(str(e), sys)

    # =====================================================
    # TRAIN MODEL
    # =====================================================
    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            logger.info("Training XGBoost model")

            model = XGBClassifier(

                random_state=self.config["training"]["random_state"],

                n_estimators=self.config["training"]["n_estimators"],

                max_depth=self.config["training"]["max_depth"],

                learning_rate=self.config["training"]["learning_rate"],

                subsample=self.config["training"]["subsample"],

                colsample_bytree=self.config["training"]["colsample_bytree"],

                gamma=self.config["training"]["gamma"],

                reg_alpha=self.config["training"]["reg_alpha"],

                reg_lambda=self.config["training"]["reg_lambda"],

                min_child_weight=self.config["training"]["min_child_weight"],

                objective=self.config["training"]["objective"],

                eval_metric=self.config["training"]["eval_metric"],

                early_stopping_rounds=self.config["training"]["early_stopping_rounds"]
            )

            model.fit(
                X_train,
                y_train,

                eval_set=[
                    (X_train, y_train),
                    (X_test, y_test)
                ],

                verbose=50
            )

            logger.info("Model training completed")

            return model

        except Exception as e:
            raise CustomException(str(e), sys)

    # =====================================================
    # COMPLETE TRAINING PIPELINE
    # =====================================================
    def run(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:

            logger.info("Model training pipeline started")

            # -----------------------------------------
            # APPLY SMOTE
            # -----------------------------------------
            X_train_resampled, y_train_resampled = self.apply_smote(
                X_train,
                y_train
            )

            # -----------------------------------------
            # TRAIN MODEL
            # -----------------------------------------
            model = self.train_model(
                X_train_resampled,
                y_train_resampled,
                X_test,
                y_test
            )

            logger.info("Model training pipeline completed successfully")

            return model

        except Exception as e:
            raise CustomException(str(e), sys)