# this file is repsonsible for transforming the data
# - how to chategorical to numerical
# - how to handle missing data
# - how to handle outliers
# - how to handle imbalanced data
# - how to handle lable encoding
# - how to handle one hot encoding
# - how to handle feature scaling
# - how to split the data
# Path: src/components/data_transformation.py

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass

from src.utils import save_object_to_pickle
from src.logger import logger
from src.exception import ExceptionHandler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessing', 'preprocessor.pkl')
    terget_column: str = 'math_score'
    train_data_path: str = os.path.join('artifacts', 'data',  'train.csv')
    test_data_path: str = os.path.join('artifacts', 'data',  'test.csv')


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def __get_data_transfromer_object(self, data_path: str = "") -> ColumnTransformer:
        """
        This method is responsible for transforming the data (Train and Test) into numerical form

        Returns:
        ColumnTransformer: Returns the column transformer object
        """

        try:
            logger.info("Initiating data transformation process")

            df = pd.read_csv(
                self.config.train_data_path) if data_path == "" else pd.read_csv(data_path)

            if (self.config.terget_column in df.columns):
                df = df.drop(columns=[self.config.terget_column])

            self.numerical_columns = [
                column for column in df.columns if df[column].dtype in ['int64', 'float64']]
            self.categorical_columns = [
                column for column in df.columns if df[column].dtype in ['object']]

            logger.info("Numerical columns: {}".format(self.numerical_columns))
            logger.info("Categorical columns: {}".format(
                self.categorical_columns))

            logger.info("Creating numerical column processing pipeline")
            self.numerical_column_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logger.info("Creating categorical column processing pipeline")
            self.categorical_column_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            self.preprocessing = ColumnTransformer(
                transformers=[
                    ('num', self.numerical_column_pipeline, self.numerical_columns),
                    ('cat', self.categorical_column_pipeline,
                     self.categorical_columns)
                ]
            )

            logger.info("Data transformation process completed")
            return self.preprocessing
        except Exception as e:
            error_message = str(ExceptionHandler(str(e)))
            logger.error(error_message)
            raise ExceptionHandler(e)

    def intiate_data_transformer(self, train_df_file: str = "", test_df_file: str = ""):
        """
        This method is responsible for transforming the data (Train and Test) into numerical form

        Returns:
        Tuple[str, str]: Returns the path of the transformed data
        """

        try:
            logger.info("Initiating data transformation process")

            train_df = pd.read_csv(
                self.config.train_data_path) if train_df_file == "" else pd.read_csv(train_df_file)
            test_df = pd.read_csv(
                self.config.test_data_path) if test_df_file == "" else pd.read_csv(test_df_file)

            logger.info("Getting the data transformation object")

            preprocessing_obj = self.__get_data_transfromer_object()

            logger.info("Terget column: {}".format(self.config.terget_column))

            input_features = train_df.drop(
                columns=[self.config.terget_column], axis=1)
            target_feature_train_df = train_df[self.config.terget_column]

            logger.info("Input features {}".format(input_features.columns))
            logger.info("Terget is {}".format(self.config.terget_column))

            input_features_test = test_df.drop(
                columns=[self.config.terget_column], axis=1)
            target_feature_test_df = test_df[self.config.terget_column]

            logger.info(
                "Transforming the train data with the preprocessing object")

            transformed_train_data = preprocessing_obj.fit_transform(
                input_features)
            transformed_test_data = preprocessing_obj.transform(
                input_features_test)

            print("Transformed train data: ", transformed_train_data.shape)
            logger.info("Transfor train data shape: {}".format(
                transformed_train_data.shape))
            train_arr = np.c_[transformed_train_data,
                              np.array(target_feature_train_df)]

            print("Transformed test data: ", transformed_test_data.shape)
            logger.info("Transfor test data shape: {}".format(
                transformed_test_data.shape))
            test_arr = np.c_[transformed_test_data,
                             np.array(target_feature_test_df)]

            logger.info("Saving the transformed data")

            os.makedirs(os.path.dirname(
                self.config.preprocessor_obj_file_path), exist_ok=True)
            save_object_to_pickle(
                self.config.preprocessor_obj_file_path, preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )
        except Exception as e:
            error_message = str(ExceptionHandler(str(e)))
            logger.error(error_message)
            raise ExceptionHandler(e)
