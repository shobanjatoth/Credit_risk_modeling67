# import sys

# from src.pipelines.training_pipeline import TrainingPipeline

# from src.logger.logger import get_logger
# from src.exceptions.exceptions import CustomException

# logger = get_logger(__name__)


# def main():
#     try:
#         logger.info("Starting Car Price Training Application")

#         pipeline = TrainingPipeline()
#         pipeline.run()

#         logger.info("Training completed successfully ")

#     except CustomException as ce:
#         logger.error(f"Custom Exception occurred: {ce}")
#         sys.exit(1)

#     except Exception as e:
#         logger.error(f"Unhandled Exception occurred: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


import sys
from src.pipelines.training_pipeline import TrainingPipeline
from src.logger.logger import get_logger
from src.exceptions.exceptions import CustomException

logger = get_logger(__name__)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline(
            config_path="config/config.yaml",
            schema_path="config/schema.yaml"
        )

        pipeline.run_pipeline()

    except Exception as e:
        logger.error(e)
        raise CustomException(str(e), sys)
