import sys
import logging
import logging.config
from pathlib import Path

# função para setup de logging
def setup_logging(log_level=logging.INFO):
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            },
        },
        'handlers': {
            'console': { # Este é o handler principal para CloudWatch Logs
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': sys.stdout, # Direciona logs para a saída padrão (stdout)
                                      # Lambda captura stdout/stderr e envia para CloudWatch
            },
     
        },
        'root': {
            # Apenas o handler 'console' deve ser usado para o root logger
            'handlers': ['console'],
            'level': log_level,
        },
    }

    # aplicação global das configurações
    try:
        logging.config.dictConfig(logging_config)

    except Exception as e:
        # Fallback para caso haja algum problema inesperado na configuração do dictConfig
        # Isso garante que você ainda terá logs básicos no CloudWatch
        print(f"Erro ao configurar o logging com dictConfig: {e}", file=sys.stderr)
        logging.basicConfig(level=log_level, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        logging.warning("Configuração de logging via dictConfig falhou. Usando basicConfig como fallback.", exc_info=True)

    # aplicação global das configurações
    logging.config.dictConfig(logging_config)
