import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import ExceptionHandler
from src.logger import logger


def save_object_to_pickle(file_path, obj):
    """
    This method is responsible for saving the object to the disk

    Args:
    obj : object : The object that needs to be saved
    file_path : str : The file path where the object needs to be saved

    Returns:
    None
    """
    try:
        logger.info("Saving object to disk {}".format(file_path))
        with open(file_path, 'wb') as fs:
            pickle.dump(obj, fs)
    except Exception as e:
        error_message = str(ExceptionHandler(str(e)))
        logger.error(error_message)
        raise ExceptionHandler(e)
