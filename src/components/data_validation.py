# import os
# import sys
# import json
# from logger.logger import get_logger
# from exceptions.exceptions import CustomException

# logger = get_logger(__name__)


# class DataValidation:

#     def __init__(self, schema, report_path):
#         self.schema = schema
#         self.expected_columns = schema["columns"]
#         self.strict = schema["strict"]
#         self.report_path = report_path

#         os.makedirs(os.path.dirname(report_path), exist_ok=True)

#     def validate(self, df):
#         try:
#             validation_status = True
#             errors = {}

#             missing = [
#                 col for col in self.expected_columns
#                 if col not in df.columns
#             ]

#             if missing:
#                 validation_status = False
#                 errors["missing_columns"] = missing

#             if self.strict:
#                 unexpected = [
#                     col for col in df.columns
#                     if col not in self.expected_columns
#                 ]

#                 if unexpected:
#                     validation_status = False
#                     errors["unexpected_columns"] = unexpected

#             for col, dtype in self.expected_columns.items():
#                 if col in df.columns:
#                     if str(df[col].dtype) != dtype:
#                         validation_status = False
#                         errors[col] = f"Expected {dtype}, got {df[col].dtype}"

#             with open(self.report_path, "w") as f:
#                 json.dump(
#                     {"validation_status": validation_status, "errors": errors},
#                     f,
#                     indent=4
#                 )

#             return validation_status

#         except Exception as e:
#             logger.error(e)
#             raise CustomException(str(e), sys)



import os
import sys
import json

from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


class DataValidation:

    def __init__(self, schema, report_path):
        self.schema = schema
        self.expected_columns = schema["columns"]
        self.strict = schema["strict"]
        self.report_path = report_path

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def validate(self, df):
        try:
            validation_status = True
            errors = {}

            missing = [
                col for col in self.expected_columns
                if col not in df.columns
            ]

            if missing:
                validation_status = False
                errors["missing_columns"] = missing

            if self.strict:
                unexpected = [
                    col for col in df.columns
                    if col not in self.expected_columns
                ]

                if unexpected:
                    validation_status = False
                    errors["unexpected_columns"] = unexpected

            for col, dtype in self.expected_columns.items():
                if col in df.columns:
                    if dtype not in str(df[col].dtype):
                        validation_status = False
                        errors[col] = f"Expected {dtype}, got {df[col].dtype}"

            with open(self.report_path, "w") as f:
                json.dump(
                    {"validation_status": validation_status, "errors": errors},
                    f,
                    indent=4
                )

            return validation_status

        except Exception as e:
            raise CustomException(str(e), sys)
