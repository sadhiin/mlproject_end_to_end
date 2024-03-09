import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from src.exception import ExceptionHandler
from src.logger import logger



def read_yaml_file(file_path):
    """
    This method is responsible for reading any yaml file.

    Args:
    file_path (str) : The file path of the yaml file.

    Returns:
    dict: Returns a dictonay for the key value pair.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except yaml.YAMLError as e:
        error_message = str(ExceptionHandler(str(e)))
        logger.error(error_message)
        raise ExceptionHandler(e)

def write_yaml_file(file_path:str, data:dict)->bool:
    """
    This method is responsible for saving the dictonary to the disk.

    Args:
    file_path (str) : The file path where the object needs to be saved.
    obj  (object) : The data that needs to be saved.

    Returns:
    bool: Returns True if the data is saved successfully.
    """
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
            logger.info(f"Data successfully written to {file_path}")
        return True
    except Exception as e:
        error_message = str(ExceptionHandler(str(e)))
        logger.error(error_message)
        raise ExceptionHandler(e)

def save_object_to_pickle(file_path: str, obj):
    """
    This method is responsible for saving the object to the disk

    Args:
    file_path (str) : The file path where the object needs to be saved
    obj  (object) : The object that needs to be saved

    Returns:
    bool: Returns True if the object is saved successfully
    """
    try:
        logger.info("Saving object to disk {}".format(file_path))
        with open(file_path, 'wb') as fs:
            pickle.dump(obj, fs)
        return True
    except Exception as e:
        error_message = str(ExceptionHandler(str(e)))
        logger.error(error_message)
        raise ExceptionHandler(e)


def load_obj(file_path: str):
    """load the object from the disk (pickle file)

    Args:
        file_path (str): file path where the object is saved.

    Returns:
        loaded object: Returns the loaded object
    """
    try:
        with open(file_path, 'rb') as fs:
            return pickle.load(fs)
    except Exception as e:
        ExceptionHandler(e)
        logger.error(f"Error: {e}")
