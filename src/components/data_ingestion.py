# this file is repsonsible for reading/loading the data
# - how to read the data
# - how to write the data
# - path: src/components/data_ingestion.py

import os
import sys
import pandas as pd
from typing import Tuple
from src.logger import logger
from dataclasses import dataclass
from src.exception import ExceptionHandler
from sklearn.model_selection import train_test_split


from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    csv_path: str = "artifacts/students.csv"
    train_data_path: str = os.path.join('artifacts', 'data',  'train.csv')
    test_data_path: str = os.path.join('artifacts', 'data',  'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data', 'raw', 'data.csv')


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def split_data(self, data: pd.DataFrame, ratio: float = 0.2) -> None:
        try:
            logger.info("Splitting data into train and test")
            train, test = train_test_split(
                data, test_size=ratio, random_state=42, shuffle=True)
            train.to_csv(self.config.train_data_path, index=False, header=True)
            test.to_csv(self.config.test_data_path, index=False, header=True)
        except Exception as e:
            error_message = str(ExceptionHandler(str(e)))
            logger.error(error_message)
            raise ExceptionHandler(e)

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading data")
            data = pd.read_csv(self.config.raw_data_path)
            return data
        except Exception as e:
            error_message = str(ExceptionHandler(str(e)))
            logger.error(error_message)
            raise ExceptionHandler(e)

    def intiate_data_ingestion(self) -> Tuple[str, str]:
        try:
            logger.info("Initiating data ingestion process")
            logger.info("Creating raw data directory")

            df = pd.read_csv(self.config.csv_path)
            os.makedirs(os.path.dirname(
                self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logger.info("Creating train data directory")
            os.makedirs(os.path.dirname(
                self.config.train_data_path), exist_ok=True)

            logger.info("Creating test data directory")
            os.makedirs(os.path.dirname(
                self.config.test_data_path), exist_ok=True)

            logger.info('Initiating data split process')
            self.split_data(df)

            logger.info("Data ingestion process completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            error_message = str(ExceptionHandler(str(e)))
            logger.error(error_message)
            raise ExceptionHandler(e)


if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    train_data, test_data = data_ingestion.intiate_data_ingestion()
    data = data_ingestion.load_data()
    print(data.head())
    print(data.shape)
    print(data.columns)
    print(data.dtypes)
    print(data.describe())
    print(data.info())
    print(data.isnull().sum())
    print(data.sample(5))

