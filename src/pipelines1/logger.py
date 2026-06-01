import logging
import os
from datetime import datetime

def get_logger(name:str):

    os.makedirs("logs",exist_ok=True)

    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formater=logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")

    file_handler=logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formater)
    
    stream_handler=logging.StreamHandler()
    stream_handler.setFormatter(formater)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger