# this file is responsible for prediction form the model.
import os
import sys
import numpy as np
import pandas as pd
from src.logger import logger
from src.utils import load_obj
from src.exception import ExceptionHandler

from dataclasses import dataclass
from typing import List

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float,
                 ) -> None:
        self.geder = gender,
        self.race_ethnicity = race_ethnicity,
        self.parental_level_of_education = parental_level_of_education,
        self.lunch = lunch,
        self.test_preparation_course = test_preparation_course,
        self.reading_score = reading_score,
        self.writing_score = writing_score

    def get_data_as_df(self):
        """This function is responsible for converting the given data into the dataframe.

        Returns:
            pandas DataFrame: Returns the dataframe of the given data.
        """
        try:
            data = {
                "gender": [self.geder],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(data)
        except Exception as e:
            ExceptionHandler(e)
            logger.error(f"Error: {e}")


@dataclass
class PredictionPipelineConfig:
    model_path: str = "artifact/models/trained_model.pkl"
    preprocessor_path: str = "artifact/preprocessor/preprocessor.pkl"


class PredictionPipeline:
    def __init__(self, cfg: PredictionPipelineConfig) -> None:
        self.config = cfg

    def predict(self, features)->list:
        """This function is responsible for predicting the target variable form the given features.

        Args:
            features (pd.dataframe): feature datafrmame processed by the CustomeData class.

        Returns:
            model_prediction: Returns the prediction from the model.
        """

        model = load_obj(file_path=self.config.model_path)
        preprocessor = load_obj(file_path=self.config.preprocessor_path)
        try:
            features = preprocessor.transform(features)
            prediction = model.predict(features)
            return prediction
        except Exception as e:
            ExceptionHandler(e)
            logger.error(f"Error: {e}")
