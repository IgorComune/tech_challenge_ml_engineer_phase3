"""AMAZON LAST MILE DELIVERY SIMULATOR (OTD)"""
import os
import numpy as np
import datetime
import pandas as pd
import simpy
from datetime import datetime
import warnings
from src.config.logging_config import setup_logging
from dotenv import load_dotenv
import mlflow
import dagshub
# ajuste dos warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

"""
Este script implementa uma simulação discreta (usando SimPy) para avaliar o impacto
de uma política de decisão corretiva no tempo final de entrega (On-Time Delivery - OTD).

A simulação integra:
1. Um modelo de Machine Learning (LightGBM, carregado via MLflow/DAGsHub) para Previsão de Tempo.
2. Uma lógica de simulação do 'Mundo Real' baseada em fatores de tráfego, clima e distância.
3. Uma Política Corretiva que ajusta o tempo de entrega com base na previsão de risco.

--------------------------------------------------------------------------------

1. ENTIDADES E VARIÁVEIS SIMULADAS

O sistema simula pedidos, cada um com as seguintes características (atributos aleatórios):
- Agent_Age, Agent_Rating: Desempenho do entregador.
- Weather, Traffic, Area, Vehicle: Condições ambientais e logísticas.
- delivery_distance: Distância real do percurso.
- is_grocery, jam_or_high_traffic: Features binárias do pedido.
- tempo_coleta: Tempo fixo de coleta no centro de distribuição.

2. RECURSOS DO SISTEMA
- Entregadores: Representado por um recurso SimPy (simulação de pool limitado).

3. FLUXO DE TEMPO E PREVISÃO

A função pedido() segue os seguintes passos de tempo:
1. Tempo de Coleta: Tempo fixo, simulado pelo 'request' do recurso Entregador.
2. Previsão de Tempo (Modelo ML):
    - O modelo LightGBM prevê o log(tempo_entrega_base) com base nos atributos do pedido.
    - O resultado é transformado de volta para minutos (np.exp()).
3. Tempo Real de Entrega:
    - Baseado em: (delivery_distance / velocidade_média) + Ruído.
    - O tempo real é uma estimativa do que realmente acontece, usado para calcular o OTD final.

4. POLÍTICA DE DECISÃO CORRETIVA

A política tenta mitigar o risco de atraso (quando PREVISÃO > 60 min).
- O algoritmo compara a previsão do ML com critérios de risco (Tráfego, Veículo, Área).
- Se a condição for atendida, um valor fixo (15-20 minutos) é subtraído do tempo real (tempo_real_total).
- O tempo corrigido (tempo_corrigido_min) é usado no yield final (tempo de entrega simulado).

5. COLETA DE MÉTRICAS (RESULTADOS)

O resultado final é armazenado na lista 'resultados_simulacao' e inclui:
- previsao_min: O tempo estimado pelo modelo ML.
- tempo_real_min: O tempo que a entrega levaria sem intervenção.
- tempo_corrigido_min: O tempo final que a entrega levou (após a intervenção da política).
- is_otd_corrigido: 1 se tempo_corrigido_min <= 120 minutos.
- decisao: A ação corretiva tomada ('Rota_Reduzida', 'Aumento_Velocidade' ou 'Nenhuma').

A meta é comparar a performance de OTD usando tempo_real_min (baseline) vs. tempo_corrigido_min (com política).

"""



# Carrega variáveis do arquivo .env
load_dotenv()


# =================================================================================
# 1. SETUP INICIAL E CARREGAMENTO DE DADOS/MODELO
# =================================================================================

# --- LIMPEZA DE VARIÁVEIS DE AMBIENTE ---
if 'MLFLOW_HOME' in os.environ:
    del os.environ['MLFLOW_HOME']
if 'MLFLOW_TRACKING_URI' in os.environ:
    del os.environ['MLFLOW_TRACKING_URI']

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_REPO = os.getenv("REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Verifica se as credenciais do Dagshub estão disponíveis antes de iniciar
if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO:
    dagshub.init(
        repo_owner=DAGSHUB_USERNAME, 
        repo_name=DAGSHUB_REPO,
        mlflow=True
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
else:
    print("Aviso: Variáveis de ambiente DAGsHub não configuradas. Pulando inicialização do MLflow/DAGsHub.")


# Variável de destino final
modelo_regressao = None 

MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
if MODEL_URI:
    MODEL_URI = MODEL_URI.strip().strip('"').strip("'")
    try:
        modelo_regressao = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Modelo '{MODEL_URI}' carregado com sucesso.")
        
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar o modelo: {e}")
else:
    print("ERRO CRÍTICO: Variável MLFLOW_MODEL_URI não encontrada.")

        
# --- VARIÁVEIS E CONSTANTES GLOBAIS DE SIMULAÇÃO ---

# Variáveis de resultados e tempo
resultados_simulacao = [] 
MINUTOS_POR_DIA = 24 * 60 # 1440 minutos

# CONSTANTE DE RESÍDUO PADRÃO 
STD_RESIDUO_PADRAO = 10 
VIÉS_CORREÇÃO_ADITIVA = 0
FATOR_DIFICULDADE = 1.05

media_pedidos_por_dia = 902
desvio_padrao_pedidos = 113

proporcao_curta_distancia = 0.55
media_curta = 5.34
desvio_curta = 2.62
proporcao_longa_distancia = 0.45
media_longa = 14.83
desvio_longa = 3.4

dist_trafego = [0.10, 0.20, 0.50, 0.20] # low, medium, high, jam
dist_clima = [0.20, 0.20, 0.20, 0.20, 0.20]
dist_veiculo = [0.587, 0.334, 0.079] # Moto, Scooter, Van
dist_weekend = [5/7, 2/7]
dist_area = [0.7487, 0.2217, 0.0262, 0.0034]
categorias_area = ['Metropolitian', 'Urban', 'Other', 'Semi-Urban']

media_rating = 4.7
desvio_padrao_rating = 0.2

# Parâmetros de tempo de coleta, baseados nos seus dados
dist_tempo_coleta = [5, 10, 15]
prob_tempo_coleta = [1347/(4010), 1332/(4010), 1331/(4010)]


# =================================================================================
# 2. FUNÇÕES DO SIMULADOR
# =================================================================================

# função de previsão
def previsao_do_modelo(modelo, dados_pedido):
    """
    Função para fazer a previsão usando seu modelo de ML.
    """
    dist_gte_10 = 1 if dados_pedido['delivery_distance'] >= 10 else 0

    dados_para_previsao = pd.DataFrame([{
        'Agent_Age': dados_pedido['Agent_Age'],
        'Agent_Rating': dados_pedido['Agent_Rating'], 
        'Weather': dados_pedido['Weather'],
        'Vehicle': dados_pedido['Vehicle'],
        'Area': dados_pedido['Area'], 
        'delivery_distance': dados_pedido['delivery_distance'], 
        'is_grocery': dados_pedido['is_grocery'],
        'jam_or_high_traffic': dados_pedido['jam_or_high_traffic'],
        'dist_gte_10': dist_gte_10,  
    }])

    if modelo is None:
        return 60.0
        
    log_tempo_previsto = modelo.predict(dados_para_previsao)
    tempo_previsto = np.exp(log_tempo_previsto)
    return tempo_previsto[0]

# função de pedido: entidade do sistema
def pedido(env, entregadores, dados_entrada, previsao):
    """Processo de um único pedido, do recebimento à entrega."""

    global resultados_simulacao
    global MINUTOS_POR_DIA 
    global STD_RESIDUO_PADRAO
    global VIÉS_CORREÇÃO_ADITIVA
    global FATOR_DIFICULDADE

    dados = dados_entrada.copy() 

    # recurso do sistema
    with entregadores.request() as req:
        yield req
        
        # tempo de coleta
        yield env.timeout(dados['tempo_coleta'])

        # --- CÁLCULO DO TEMPO REAL DE ENTREGA ---
        
        # calculo do tempo real de viagem
        tempo_real_viagem = np.random.normal(previsao*FATOR_DIFICULDADE, STD_RESIDUO_PADRAO) + VIÉS_CORREÇÃO_ADITIVA
        tempo_real_viagem = max(0, tempo_real_viagem)
        
        
        # --- POLÍTICA DE DECISÃO CORRETIVA ---
        tempo_corrigido_viagem = max(0, tempo_real_viagem)
        decisao = 'Nenhuma'
        
        # Variáveis para a política corretiva
        desconto_base = 20 
        desconto_agressivo = 30 
        desconto_risco_extremo = 40 # desconto máximo

        # Fatores de Risco
        agente_mais_velho = dados['Agent_Age'] > 30 
        clima_ruim = dados['Weather'] != 'Sunny'
        dist_gte_10_logica = 1 if dados['delivery_distance'] >= 10 else 0 

        # trigger da política corretiva
        if previsao >= 40:
            
            # 1. Risco Extremo (combinação de tráfego e distância)
            if dados['jam_or_high_traffic'] == 1 and dados['delivery_distance'] >= 15:
                
                tempo_corrigido_viagem = max(0, tempo_real_viagem - desconto_risco_extremo) 
                decisao = 'Intervencao_Risco_Extremo'
            
            # 2. Alto Risco (Tráfego Alto/Jam e distância)
            elif dados['jam_or_high_traffic'] == 1 and dados['delivery_distance'] <15:
                
                if dados['Vehicle'] in ['Scooter', 'Moto']:
                    
                    if agente_mais_velho:
                        tempo_corrigido_viagem = max(0, tempo_real_viagem - desconto_risco_extremo)
                        decisao = 'Intervenção_Alto_Reforçado'
                    else:
                        tempo_corrigido_viagem = max(0, tempo_real_viagem - desconto_agressivo) 
                        decisao = 'Intervenção_Alto_Risco'

            # 3. Médio Risco (demais condições de tráfego )
            elif dados['jam_or_high_traffic']==0: 
                if dados['Vehicle'] in ['Scooter', 'Moto']:
                    tempo_corrigido_viagem = max(0, tempo_real_viagem - desconto_base)
                    decisao = 'Intervenção_Médio_Risco'

        
        # 4. TEMPO DE ENTREGA E CONCLUSÃO
        yield env.timeout(max(0, tempo_corrigido_viagem))
        
        
        # --- CÁLCULOS FINAIS PARA LOG ---
        tempo_real_total = dados['tempo_coleta'] + max(0, tempo_real_viagem)
        tempo_corrigido_total = dados['tempo_coleta'] + max(0, tempo_corrigido_viagem)
        is_otd_corrigido = 1 if (tempo_corrigido_total) <= 120 else 0

        # persistência do resultado do pedido 
        resultados_simulacao.append({
            'id_pedido': dados['id_pedido'], 'previsao_min': previsao,
            'tempo_real_min': tempo_real_total, 'tempo_corrigido_min': tempo_corrigido_total,
            'delivery_distance': dados['delivery_distance'], 'tempo_coleta': dados['tempo_coleta'],
            'Agent_Rating': dados['Agent_Rating'],'Agent_Age': dados['Agent_Age'],
            'Vehicle': dados['Vehicle'],
            'Weather': dados['Weather'], 'Traffic': dados['Traffic'],
            'Area': dados['Area'], 'weekend': dados['weekend'],
            'is_grocery': dados['is_grocery'],
            'jam_or_high_traffic': dados['jam_or_high_traffic'],
            'is_sunny_weather': dados['is_sunny_weather'],
            'tempo_criacao_sim': dados['tempo_criacao'],
            'tempo_conclusao_sim': env.now, 'decisao': decisao,
            'is_otd_corrigido': is_otd_corrigido, 'dia': int(np.ceil(dados['tempo_criacao'] / MINUTOS_POR_DIA))
        })

# =================================================================================
# 3. GERADOR DE PEDIDOS (MANTIDO)
# =================================================================================

def gerador_de_pedidos(env, pool_entregadores, modelo, prob_grocery_desejada):
    
    # === DECLARAÇÃO GLOBAL ===
    global resultados_simulacao 
    global media_pedidos_por_dia
    global desvio_padrao_pedidos
    global media_curta
    global proporcao_curta_distancia
    global media_longa
    global desvio_curta
    global desvio_longa
    global dist_trafego
    global dist_clima
    global dist_weekend
    global dist_veiculo
    global media_rating
    global desvio_padrao_rating
    global categorias_area
    global dist_area
    global dist_tempo_coleta
    global prob_tempo_coleta
    global MINUTOS_POR_DIA 

    # dias de simulação
    num_dias_simulacao = 7
    
    indice_ultimo_pedido_calculado = 0 
    
    for dia in range(1, num_dias_simulacao + 1):
        
        print(f"\n--- INICIANDO SIMULAÇÃO DO DIA {dia} (Tempo Simulado: {env.now:.0f} min) ---")

        # pedidos diários
        pedidos_hoje = int(np.round(np.random.normal(media_pedidos_por_dia, desvio_padrao_pedidos)))
        if pedidos_hoje <= 0: pedidos_hoje = 1
        
        media_inter_chegada = MINUTOS_POR_DIA / pedidos_hoje 
        
        for i in range(pedidos_hoje):
            
            # CORREÇÃO DE TRAVAMENTO NO YIELD
            if i == 0:
                tempo_espera = 0
            else:
                tempo_espera = np.random.exponential(media_inter_chegada)
            
            if not isinstance(tempo_espera, (int, float)) or tempo_espera <= 0 or not np.isfinite(tempo_espera):
                tempo_espera = 0.01 
                
            yield env.timeout(tempo_espera)
            
            # --- GERAÇÃO DE FEATURES NO GERADOR DE PEDIDO ---
            delivery_distance = np.random.normal(
                media_curta if np.random.random() < proporcao_curta_distancia else media_longa,
                desvio_curta if np.random.random() < proporcao_curta_distancia else desvio_longa
            )
            if delivery_distance < 0: delivery_distance = 0
            
            is_grocery = np.random.choice([1, 0], p=[prob_grocery_desejada, 1 - prob_grocery_desejada]) 
            
            Traffic = np.random.choice(['Low', 'Medium', 'High', 'Jam'], p=dist_trafego)
            Weather = np.random.choice(['Stormy', 'Sunny', 'Foggy', 'Windy', 'Sandstorms'], p=dist_clima)
            weekend = np.random.choice([False, True], p=dist_weekend)
            vehicle_raw = np.random.choice(['Moto', 'Scooter', 'Van'], p=dist_veiculo)
            Agent_Rating = np.random.normal(media_rating, desvio_padrao_rating) 
            Area = np.random.choice(categorias_area, p=dist_area)
            jam_or_high_traffic = 1 if Traffic in ['High', 'Jam'] else 0
            is_sunny_weather = 1 if Weather == 'Sunny' else 0
            Agent_Age = int(np.random.normal(27, 5.76)) 

            dados_pedido = {
                'id_pedido': f'pedido_{env.now:.4f}_{i}',
                'Agent_Rating': Agent_Rating, 'Weather': Weather, 'Traffic': Traffic,
                'Vehicle': vehicle_raw, 
                'Area': Area, 'weekend': weekend,
                'delivery_distance': delivery_distance, 'is_grocery': is_grocery,
                'jam_or_high_traffic': jam_or_high_traffic,
                'is_sunny_weather': is_sunny_weather,
                'Agent_Age': Agent_Age, 
                'tempo_coleta': np.random.choice(dist_tempo_coleta, p=prob_tempo_coleta),
                'tempo_criacao': env.now
            }
            
            previsao = previsao_do_modelo(modelo, dados_pedido)
            env.process(pedido(env, pool_entregadores, dados_pedido, previsao))
            
        
        tempo_restante_no_ciclo = (dia * MINUTOS_POR_DIA) - env.now
        if tempo_restante_no_ciclo > 0:
            yield env.timeout(tempo_restante_no_ciclo)
        
        # verificação do encerramento da simulação
        if len(resultados_simulacao) > indice_ultimo_pedido_calculado:
            
            df_pedidos_concluidos_no_dia = pd.DataFrame(
                resultados_simulacao[indice_ultimo_pedido_calculado:]
            )
            
            tempo_limite_diario = dia * MINUTOS_POR_DIA
            
            pedidos_concluidos_no_ciclo = df_pedidos_concluidos_no_dia[
                df_pedidos_concluidos_no_dia['tempo_conclusao_sim'] <= tempo_limite_diario
            ]
            
            if not pedidos_concluidos_no_ciclo.empty:
                otd_diario_corrigido = pedidos_concluidos_no_ciclo['is_otd_corrigido'].mean()
                
                logger.info(f"--- FIM DO DIA {dia} ---")
                logger.info(f"  > OTD do Dia (Tempo Corrigido <= 120 min): {otd_diario_corrigido:.2%}")
                logger.info(f"  > Total de Pedidos Concluídos no Dia: {len(pedidos_concluidos_no_ciclo)}")
            
            indice_ultimo_pedido_calculado = len(resultados_simulacao)
        else:
            pass
            logger.info(f"--- FIM DO DIA {dia} --- Nenhum pedido concluído neste ciclo. Tempo atual: {env.now:.0f} min.")


    yield env.timeout(0)
    
# =================================================================================
# 4. EXECUÇÃO
# =================================================================================

if __name__ == "__main__":
    print("Iniciando a simulação.")

    # configuração do logging
    logger = setup_logging()
    
    # configuração do ambiente
    env = simpy.Environment()
    entregadores = simpy.Resource(env, capacity=800)
    
    PROB_GROCERY_DESEJADA = 0.2
    
    # início do processo de geração de pedidos
    env.process(gerador_de_pedidos(env, entregadores, modelo_regressao, PROB_GROCERY_DESEJADA))

    # simulacao em minutos .
    MINUTOS_COBERTURA = 24 * 60
    DIAS_SIMULACAO = 7
    env.run(until=(DIAS_SIMULACAO * MINUTOS_POR_DIA) + MINUTOS_COBERTURA)

    
    # resultados da simulação
    df_resultados_finais = pd.DataFrame(resultados_simulacao)
    
    print("\n--- Resultados Finais da Simulação ---")
    logger.info(df_resultados_finais.head())
    

    prop_sucesso = (df_resultados_finais['tempo_corrigido_min'] <= 120).mean()
    logger.info(f"\nProporção de pedidos com tempo de entrega corrigido <= 120 minutos: {prop_sucesso:.2%}")

    nome_arquivo = f'data/simulation/sim_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df_resultados_finais.to_csv(nome_arquivo, index=False)
    logger.info(f"\nResultados salvos em: {nome_arquivo}")