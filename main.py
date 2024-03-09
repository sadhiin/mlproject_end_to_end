import os
import sys
import argparse
from src.logger import logger
from src.exception import ExceptionHandler
from src.pipeline.train_pipeline import TrainingPipeline

from threading import Thread

if __name__=="__main__":
    logger.info(f"Runing 'main.py' to train the model")
    
    parser = argparse.ArgumentParser(description="Train the model base on for prediction")
    parser.add_argument("-cfg","--configuration",type=str, help="A yaml file that contain the configureation about the required files.", default='config.yaml')
    parser.add_argument("-m","--models", type=str, help="Yaml file of models to be trained on", default='models.yaml')
    parser.add_argument("-p", "--params",type=str, help="Yaml file of models parameter for hyperparameter tuning", default='params.yaml')
    
    args = parser.parse_args()
    
    if args.configuration and args.models and args.params:
        logger.info(f"Using configuration from '{args.configuration}' file.")
        logger.info(f"Using model informations from '{args.models}' file.")
        logger.info(f"Using model parameters from '{args.params}' file.")
        
        train = TrainingPipeline(args.configuration, args.models, args.params)
        logger.info("Stratring the training.")
        trainer_tread = Thread(train.run())
        trainer_tread.daemon = True
        trainer_tread.start()
        logger.info("Training done!")
    else:
        raise ValueError("No configuration files are passed. Those are required to train the model.")