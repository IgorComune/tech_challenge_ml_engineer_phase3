"""Arquivo de simulacao de ações de melhoria do otd a partir da previsão do modelo ML"""
import numpy as np
import datetime
import pandas as pd
import simpy
from datetime import datetime
import joblib

# entidades

#PEDIDOS:
    # pct_pedidos_groceries
    # pct_pedidos_other
        
        # atributos aleatorios:
            # distancia: distr normal ou exponencial
            # qtde_pedidos: medir ou identificar
            # tempo_coleta
            # tempo_entrega_base: calculado na distancia, trafego e velocidade média do veiculo
            # previsao: resultado do modelo
            # fator trafego
            # fator clima
# ENTREGADOR


# recursos:

        # pool de entregadores

# decisoes corretivas:


        # previsao + desvio > = 120 and traffic_jam:
        #       # acao: aumento velocidade:  
                # correcao: - 5
                # tempo_base: (distancia * velocidade)* fator_clima * fator_trafego
                # tempo_corrigido = tempo - correcao

        # previsao + desvio > = 120 and traffic_jam or traffic_:
    

#O Tempo Real da Entrega (real e real - correcao):

#O tempo de entrega real será a soma do tempo de coleta mais o tempo de entrega ajustado.

#Tempo Real (real): coleta_tempo + entrega_tempo_base.

#Tempo Real com Correção (real - correcao): coleta_tempo + (entrega_tempo_base - valor de correção). Este é o valor que reflete o efeito da sua ação.

#Coleta de Métricas:

#O Salabim permite coletar e armazenar dados de cada entidade. Você deve registrar as seguintes informações para cada pedido:

#tempo_total_previsto

#tempo_real (sem correção)

#tempo_real_com_correcao (o valor que você quer simular)

#categoria

# Ao final da simulação, você terá uma base de dados para comparar os resultados.


# =================================================================================
# 1. SETUP INICIAL E CARREGAMENTO DE DADOS/MODELO
# =================================================================================

# Carregar o modelo treinado
try:
    modelo_regressao = joblib.load('src/models/modelo_teste.pkl')
    print("Modelo de ML carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_teste.pkl' não encontrado. Verifique o caminho.")
    exit()

# Suas variáveis e distribuições de simulação
media_pedidos_por_dia = 902
desvio_padrao_pedidos = 113

proporcao_curta_distancia = 0.55
media_curta = 5.34
desvio_curta = 2.62
proporcao_longa_distancia = 0.45
media_longa = 14.83
desvio_longa = 3.4

dist_is_grocery = [0.06, 0.94]
dist_trafego = [0.10, 0.20, 0.50, 0.20] # low, medium, high, jam
dist_clima = [0.20, 0.20, 0.20, 0.20, 0.20]
dist_veiculo = [0.587, 0.334, 0.079] # Moto, Scooter, Van
dist_weekend = [5/7, 2/7]
dist_area = [0.7487, 0.2217, 0.0262, 0.0034]
categorias_area = ['Metropolitian', 'Urban', 'Other', 'Semi-Urban']

media_rating = 4.7
desvio_padrao_rating = 0.2

media_geral_grocery = 36.56
std_geral_grocery = 10.26

# Parâmetros de tempo de coleta, baseados nos seus dados
dist_tempo_coleta = [5, 10, 15]
prob_tempo_coleta = [1347/(1347+1332+1331), 1332/(1347+1332+1331), 1331/(1347+1332+1331)]

# NOVO: Dicionário para velocidades médias baseado em área, tráfego e veículo
dados_velocidade_por_condicao = {
    'Metropolitian': {
        'High': {'motorcycle': 1.815406, 'scooter': 2.012594, 'van': 2.020903},
        'Jam': {'motorcycle': 4.339058, 'scooter': 4.917653, 'van': 4.934396},
        'Low': {'motorcycle': 4.307250, 'scooter': 4.866703, 'van': 4.907279},
        'Medium': {'motorcycle': 4.681098, 'scooter': 5.253084, 'van': 5.213391}
    },
    'Other': {
        'High': {'motorcycle': 2.042172, 'scooter': 2.613050, 'van': 2.055367},
        'Jam': {'motorcycle': 5.227387, 'scooter': 5.885592, 'van': 6.040598},
        'Low': {'motorcycle': 4.800772, 'scooter': 5.444634, 'van': 5.658412},
        'Medium': {'motorcycle': 5.035232, 'scooter': 5.841980, 'van': 4.089709}
    },
    'Semi-Urban': {
        'High': {'motorcycle': 1.152298},
        'Jam': {'motorcycle': 3.258818, 'scooter': 4.067922, 'van': 2.904431},
        'Medium': {'motorcycle': 3.274100}
    },
    'Urban': {
        'High': {'motorcycle': 1.992608, 'scooter': 2.353110, 'van': 2.331909},
        'Jam': {'motorcycle': 4.681708, 'scooter': 5.549280, 'van': 5.552400},
        'Low': {'motorcycle': 4.686939, 'scooter': 4.872659, 'van': 5.100601},
        'Medium': {'motorcycle': 5.000355, 'scooter': 5.740665, 'van': 5.420697}
    }
}


# Lista para armazenar os resultados da simulação
resultados_simulacao = []

# =================================================================================
# 2. FUNÇÕES DO SIMULADOR
# =================================================================================

def previsao_do_modelo(modelo, dados_pedido):
    """
    Função para fazer a previsão usando seu modelo de ML.
    """
    dados_para_previsao = pd.DataFrame([{
        'Agent_Rating': dados_pedido['Agent_Rating'], 
        'Weather': dados_pedido['Weather'],
        'Traffic': dados_pedido['Traffic'], 
        'Vehicle': dados_pedido['Vehicle'],
        'Area': dados_pedido['Area'], 
        'weekend': dados_pedido['weekend'],
        'delivery_distance': dados_pedido['delivery_distance'], 
        'is_grocery': dados_pedido['is_grocery'],
        'jam_or_high_traffic': dados_pedido['jam_or_high_traffic'],
        'is_sunny_weather': dados_pedido['is_sunny_weather']
    }])
    tempo_previsto = modelo.predict(dados_para_previsao)
    return tempo_previsto[0]

def pedido(env, entregadores, dados, previsao):
    """Processo de um único pedido, do recebimento à entrega."""
    
    with entregadores.request() as req:
        yield req
        yield env.timeout(dados['tempo_coleta'])

    #if dados['is_grocery'] == 0:
        # NOVO: Cálculo do tempo real baseado em distância e velocidade por condição
        area = dados.get('Area')
        traffic = dados.get('Traffic')
        vehicle = dados.get('Vehicle')
        
        # Tenta encontrar a velocidade na nova tabela, se não encontrar, usa 
        # a previsão como um fallback (adicionando um pouco de ruído)
        try:
            velocidade_media = dados_velocidade_por_condicao.get(area, {}).get(traffic, {}).get(vehicle, None)
            if velocidade_media is not None and velocidade_media > 0:
                tempo_estimado_viagem = (dados['delivery_distance'] / velocidade_media) * 60
                tempo_real_total = np.random.normal(tempo_estimado_viagem, tempo_estimado_viagem * 0.1)
            else:
                tempo_real_total = previsao + np.random.normal(0, 10)
        except KeyError:
                tempo_real_total = previsao + np.random.normal(0, 10)
        #else:
         #   tempo_real_total = np.random.normal(media_geral_grocery, std_geral_grocery)
        
        # Subtrai o tempo de coleta para obter apenas o tempo de viagem
        tempo_real_total = max(0, tempo_real_total - dados['tempo_coleta'])
        
        tempo_corrigido = tempo_real_total
        decisao = 'Nenhuma'

        # política: Correção para qualquer pedido acima de 45 minutos
        if previsao > 60:
            if dados['Traffic'] in ['High', 'Jam']:
                if dados['Vehicle'] in ['Scooter', 'Moto'] and dados['Area'] in ['Urban', 'Metropolitian']:
                    tempo_corrigido = max(0, tempo_real_total - 15)
                    decisao = 'Aumento_Velocidade'
                elif dados['Vehicle'] in ['Scooter', 'Moto'] and dados['Area'] in ['Semi-Urban', 'Other']:
                    tempo_corrigido = max(0, tempo_real_total - 20)
                    decisao = 'Rota_Reduzida'
            elif dados['Traffic'] == 'Medium':
                if dados['Vehicle'] in ['Scooter', 'Moto']:
                    tempo_corrigido = max(0, tempo_real_total - 15)
                    decisao = 'Rota_Reduzida'

        # O yield já estava correto, usa o tempo de viagem corrigido
        yield env.timeout(max(0, tempo_corrigido))
        
        resultados_simulacao.append({
            'id_pedido': dados['id_pedido'], 'previsao_min': previsao,
            'tempo_real_min': tempo_real_total, 'tempo_corrigido_min': tempo_corrigido,
            'delivery_distance': dados['delivery_distance'], 'tempo_coleta': dados['tempo_coleta'],
            'Agent_Rating': dados['Agent_Rating'], 'Vehicle': dados['Vehicle'],
            'Weather': dados['Weather'], 'Traffic': dados['Traffic'],
            'Area': dados['Area'], 'weekend': dados['weekend'],
            'is_grocery': dados['is_grocery'],
            'jam_or_high_traffic': dados['jam_or_high_traffic'],
            'is_sunny_weather': dados['is_sunny_weather'],
            'tempo_criacao_sim': dados['tempo_criacao'],
            'tempo_conclusao_sim': env.now, 'decisao': decisao
        })

# =================================================================================
# 3. GERADOR DE PEDIDOS
# =================================================================================

def gerador_de_pedidos(env, entregadores, modelo):
    """
    Gera pedidos com base nos parâmetros definidos, simulando a chegada ao longo do tempo.
    """
    num_dias_simulacao = 7
    total_pedidos_simulacao = 0
    
    for dia in range(num_dias_simulacao):
        # Gera o número de pedidos para este dia, com base na média e desvio padrão
        pedidos_hoje = int(np.round(np.random.normal(media_pedidos_por_dia, desvio_padrao_pedidos)))
        if pedidos_hoje < 0: pedidos_hoje = 0
        total_pedidos_simulacao += pedidos_hoje

        # Define a taxa de chegada para o dia
        media_inter_chegada = (24 * 60) / pedidos_hoje if pedidos_hoje > 0 else 0
        
        for i in range(pedidos_hoje):
            # 1. Simula o tempo até a chegada do próximo pedido
            yield env.timeout(np.random.exponential(media_inter_chegada))
            
            # 2. Geração dos atributos do pedido de forma aleatória
            delivery_distance = np.random.normal(
                media_curta if np.random.random() < proporcao_curta_distancia else media_longa,
                desvio_curta if np.random.random() < proporcao_curta_distancia else desvio_longa
            )
            if delivery_distance < 0: delivery_distance = 0
            
            is_grocery = 1 if np.random.choice(['Grocery', 'Outros'], p=dist_is_grocery) == 'Grocery' else 0
            Traffic = np.random.choice(['Low', 'Medium', 'High', 'Jam'], p=dist_trafego)
            Weather = np.random.choice(['Stormy', 'Sunny', 'Foggy', 'Windy', 'Sandstorms'], p=dist_clima)
            weekend = np.random.choice([False, True], p=dist_weekend)
            Vehicle = np.random.choice(['Moto', 'Scooter', 'Van'], p=dist_veiculo)
            Agent_Rating = np.clip(np.random.normal(media_rating, desvio_padrao_rating), 3.5, 5.0)
            Area = np.random.choice(categorias_area, p=dist_area)
            jam_or_high_traffic = 1 if Traffic in ['High', 'Jam'] else 0
            is_sunny_weather = 1 if Weather == 'Sunny' else 0

            dados_pedido = {
                'id_pedido': f'pedido_{env.now}_{i}',
                'Agent_Rating': Agent_Rating, 'Weather': Weather, 'Traffic': Traffic,
                'Vehicle': Vehicle, 'Area': Area, 'weekend': weekend,
                'delivery_distance': delivery_distance, 'is_grocery': is_grocery,
                'jam_or_high_traffic': jam_or_high_traffic,
                'is_sunny_weather': is_sunny_weather,
                'tempo_coleta': np.random.choice(dist_tempo_coleta, p=prob_tempo_coleta),
                'tempo_criacao': env.now
            }
            
            previsao = previsao_do_modelo(modelo, dados_pedido)
            env.process(pedido(env, entregadores, dados_pedido, previsao))
    
    print(f"Total de pedidos gerados na simulação: {total_pedidos_simulacao}")
    yield env.timeout(0)

# =================================================================================
# 4. EXECUÇÃO
# =================================================================================

if __name__ == "__main__":
    print("Iniciando a simulação completa...")
    
    # Configura o ambiente
    env = simpy.Environment()
    entregadores = simpy.Resource(env, capacity=800)
    
    # Inicia o processo de geração de pedidos
    env.process(gerador_de_pedidos(env, entregadores, modelo_regressao))

    # Executa a simulação por 7 dias (7 * 24 * 60 minutos)
    env.run(until=7 * 24 * 60)
    
    # Análise dos resultados
    df_resultados_finais = pd.DataFrame(resultados_simulacao)
    
    print("\n--- Resultados Finais da Simulação ---")
    print(df_resultados_finais.head())
    
    prop_sucesso = (df_resultados_finais['tempo_corrigido_min'] <= 120).mean()
    print(f"\nProporção de pedidos com tempo de entrega corrigido <= 120 minutos: {prop_sucesso:.2%}")

    # Salva os resultados em um arquivo
    nome_arquivo = f'data/simulation/sim_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df_resultados_finais.to_csv(nome_arquivo, index=False)
    print(f"\nResultados salvos em: {nome_arquivo}")