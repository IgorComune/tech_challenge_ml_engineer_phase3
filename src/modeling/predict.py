"""Aqruivo de funções de predição do modelo"""
from pandas import DataFrame

def predict(data_input: DataFrame, model=None) -> list:
    """
    Recebe os dados de entrada, executa o pré-processamento
    e retorna a previsão de duração do ciclo de entrega.
    """
    if data_input.empty:
        raise ValueError("A entrada de dados não pode ser vazia.")

    output = model.predict(data_input)
    return output.tolist()