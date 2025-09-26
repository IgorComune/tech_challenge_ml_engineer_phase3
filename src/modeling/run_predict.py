"""Arquivo de predição do modelo"""
import os
import pandas as pd
import pickle
import logging
from src.modeling.predict import predict
from src.config.logging_config import setup_logging


# configurações de logging
logger = setup_logging() 

# path do modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pred_model.pkl')

# carregamento do modelo
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


# dataset de predição

data = 
# predicao do modelo

resultado = predict(data, model=model)
logger.info(resultado)



