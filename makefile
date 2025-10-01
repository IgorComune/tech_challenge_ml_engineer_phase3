#################################################################################
# GLOBALS (CORRIGIDO PARA POWERSHELL)                                           #
#################################################################################

# --- Variável Chave: Força o uso do Powershell para comandos complexos ---
SHELL := powershell.exe

PROJECT_NAME = src
PYTHON_INTERPRETER = python

# --- Caminhos dos Scripts ---
TRAIN_SCRIPT = $(PROJECT_NAME)/modeling/train.py
PREDICT_SCRIPT = $(PROJECT_NAME)/run_predict.py

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Instala as dependências Python
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -e .

## Deleta todos os arquivos Python compilados e caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint usando flake8, black, e isort
.PHONY: lint
lint:
	flake8 $(PROJECT_NAME)
	isort --check --diff $(PROJECT_NAME)
	black --check $(PROJECT_NAME)

## Formata código fonte com black e isort
.PHONY: format
format:
	isort $(PROJECT_NAME)
	black $(PROJECT_NAME)

#################################################################################
# PROJECT RULES (MLOPS)                                                         #
#################################################################################

## Processa dados brutos para formato intermediário
.PHONY: data
data:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/scripts/make_dataset.py --input_filepath data/raw --output_filepath data/interim

## Treina o modelo, ajusta hiperparâmetros e loga no MLflow/DagsHub
.PHONY: train
train:
	@echo "Iniciando pipeline de treino e rastreamento com MLflow..."
	python -m src.modeling.run_train



## Executa o script de predição (run_predict)
.PHONY: predict
predict:
	@echo "Executando predição do modelo..."
	python -m src.modeling.run_predict

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
