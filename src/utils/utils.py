import yaml
import sys

from src.exceptions.exceptions import CustomException


def load_yaml(path: str):
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise CustomException(str(e), sys)

