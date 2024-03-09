# this file is responsible for training the model
# - train the model

import os
import sys
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

class TrainingPipeline:
    def __init__(self, cfg_path, model_cfg, params_cfg)->None:
        self.cfg= cfg_path
        self.models = model_cfg
        self.params = params_cfg
    
    def run(self)->None:
        
        data_ingestion_config = DataIngestionConfig.from_yaml(self.cfg)
        data_ingestion_obj = DataIngestion(data_ingestion_config)
        train_data, test_data = data_ingestion_obj.intiate_data_ingestion()

        data_transformation_config = DataTransformationConfig.from_yaml(self.cfg)
        data_transformation_obj = DataTransformation(data_transformation_config)

        train_arr,test_arr, _ = data_transformation_obj.intiate_data_transformer(
            test_df_file=test_data, 
            train_df_file=train_data)

        model_trainer_config = ModelTrainerConfig.from_yaml(self.cfg)
        model_trainer = ModelTrainer(model_trainer_config)

        print(model_trainer.initiate_model_trainer(train_arr, test_arr, self.models, self.params))
        
if __name__=="__main__":
    trainer = TrainingPipeline()
    trainer.run()