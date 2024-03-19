# this file is responsible for training the model
# - how to train the model
# - how many model i want to train
# - how to validate the model
# - writing code for model evaluation metrics (accuracy, precision, recall, r2, f1-score, auc-roc)
# - how to test the model

import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Tuple
from src.logger import logger
from src.utils import save_object_to_pickle, read_yaml_file, write_yaml_file
from src.exception import ExceptionHandler

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'models', 'trained_model.pkl')
    
    @classmethod
    def from_yaml(cls, file_path:str):
        with open(file_path, 'r') as file:
            try:
                config_data = yaml.safe_load(file)
                return cls(**config_data.get('model_trainer', {}))
            except yaml.YAMLError as e:
                logger.error(f"Error reading YAML file: {e}")
                raise ExceptionHandler(e)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        # dictonary of the models
        self.config.models_list = {
                "logistic_regression": LogisticRegression(),
                "knn": KNeighborsRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "random_forest": RandomForestRegressor(),
                "adaboost": AdaBoostRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "xgboost": XGBRegressor(),
                "catboost": CatBoostClassifier()
            }


    def __train_models(self, X_train, y_train, X_test, y_test, models_dict, params_dict) -> dict:
        """
        Trains multiple models using grid search for hyperparameter tuning and evaluates their performance.
        And Save the best model

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target values.
            X_test (np.ndarray): Testing data features.
            y_test (np.ndarray): Testing data target values.
            models_dict (dict): Dictionary containing model classes as keys and their names as values (e.g., {"LogisticRegression": "Logistic Regression"}).
            params_dict (dict): Dictionary containing grid search parameters for each model in models_dict.

        Returns:
            dict: Dictionary containing test scores for each model.
        """
        
        model_report = {}
        
        print("Printing models ", models_dict)
        
        for model_name, model_class in models_dict.items():
            try:
                logger.info(f"Running traning for {model_name}.")
                # Create model instance
                model = model_class

                # Perform grid search with cross-validation
                for CV in [3,5,7,9,10,11]:
                    gs = GridSearchCV(model, params_dict[model_name], cv=CV)
                    gs.fit(X_train, y_train)

                    # Set best parameters on the model
                    model.set_params(**gs.best_params_)
                    
                    # write_yaml_file(f"{model_name}.yaml", {'best_params':gs.best_params_, 'score':gs.best_score_})
                    
                    # Train the model with best parameters
                    model.fit(X_train, y_train)

                    # Make predictions on test data
                    y_pred = model.predict(X_test)

                    # Calculate R-squared score
                    model_report[model_name] = [
                        r2_score(y_test, y_pred),
                        gs.best_params_, CV]

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                raise ExceptionHandler(e)

        return model_report

    def initiate_model_trainer(self, train_array, test_array, params_cfg:dict) -> Tuple[str, list[float,dict]]:
        """
        Args:
            train_array (np.ndarray): Independent and dependent variables for training after preprocessed by columntransformer.
            test_array (np.ndarray): Independent and dependent variables for testing after preprocessed by columntransformer.

        Raises:
            ExceptionHandler: Handles exceptions raised during model training.

        Returns:
            r2score (float): R-squared score of the best model.
        """
        
        try:
            logger.info("Initiating model training process")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            models_list = self.config.models_list
            grid_params = params_cfg

            model_report = self.__train_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models_dict=models_list,
                params_dict=grid_params)

            # Get the best model
            best_model = max([(model, score) for model, score in model_report.items()], key=lambda x: x[1][0])
            write_yaml_file("best_model.yaml", best_model)
            
            if best_model[1][0] < 0.6:
                logger.error("Model R2 score is less than 0.6")
                raise ExceptionHandler("Model R2 score is less than 0.6. Not finding a good model")
            logger.info(f"Best model: {best_model[0]}, R2 score: {best_model[1][0]}, Best parameters: {best_model[1][1]}")

            # Save the best model
            best_model_obj = models_list[best_model[0]]
            best_model_obj.set_params(**best_model[1][1])
            best_model_obj.fit(X_train, y_train)

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)

            save_object_to_pickle(self.config.trained_model_file_path, best_model_obj)

            return best_model
        except Exception as e:
            logger.error("Error occured while training the model: {}".format(e))
            raise ExceptionHandler(e)

if __name__=="__main__":
    pass