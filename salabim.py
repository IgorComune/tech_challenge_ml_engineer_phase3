"""Arquivo de simulacao de ações de melhoria do otd a partir da previsão do modelo ML"""
import salabim as sim
import pickle as pkl
import numpy as np
import datetime
import pandas as pd



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


# --- ATRIBUTOS DA SIMULAÇÃO ---
media_pedidos_por_dia = 709.98
desvio_padrao_pedidos = 377.22

proporcao_curta_distancia = 0.55
media_curta = 5.3
desvio_curta = 0.5
proporcao_longa_distancia = 0.45
media_longa = 15.0
desvio_longa = 0.5

dist_is_grocery = [0.05, 0.95]
dist_trafego = [0.10, 0.20, 0.50, 0.20] # low, medium, high, jam
dist_clima = [0.20, 0.20, 0.20, 0.20, 0.20]
dist_veiculo = [0.587, 0.334, 0.079] # Moto, Scooter, Van

fator_clima = {'Stormy': 1.2, 'Sunny': 1.0, 'Foggy': 1.1, 'Windy': 1.05, 'Sandstorms': 1.15}
fator_trafego = {'Low': 1.0, 'Medium': 1.1, 'High': 1.2, 'Jam': 1.5}
fator_veiculo = {'Moto': 1.0, 'Scooter': 1.05, 'Van': 1.15}
fator_rating_impacto = {'Excelente': 0.9, 'Bom': 1.0, 'Regular': 1.1}

media_rating = 4.7
desvio_padrao_rating = 0.2

dist_weekend = [5/7, 2/7]

dist_area = [0.7487, 0.2217, 0.0262, 0.0034]
categorias_area = ['AreaMetropolitian', 'Urban', 'Other', 'Semi-Urban']

# --- FUNÇÃO DO MODELO DE PREVISÃO ---
def previsao_do_modelo(dados_pedido):
    """
    Função para fazer a previsão usando seu modelo de ML.
    Você deve integrar seu modelo aqui.
    """
    # Converter o dicionário de dados em um DataFrame para o modelo
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

    with open('src/models/modelo_teste.pkl', 'rb') as f:
        modelo = pkl.load(f)
    # Substitua o cálculo abaixo pela chamada do seu modelo
    # tempo_previsto = modelo.predict(dados_para_previsao)
    tempo_previsto = modelo.predict(dados_para_previsao)
    return tempo_previsto


# --- ESTRUTURA DO SIMULADOR ---
resultados_simulacao = []


class GeradorDePedidos(sim.Component):
    def process(self):
        while True:
            pedidos_hoje = int(np.round(np.random.normal(media_pedidos_por_dia, desvio_padrao_pedidos)))
            if pedidos_hoje < 0: pedidos_hoje = 0
            
            for i in range(pedidos_hoje):
                # 1. Geração dos atributos do pedido
                delivery_distance = np.random.normal(
                    media_curta if np.random.random() < proporcao_curta_distancia else media_longa,
                    desvio_curta if np.random.random() < proporcao_curta_distancia else desvio_longa
                )
                if delivery_distance < 0: delivery_distance = 0
                    
                is_grocery_cat = np.random.choice(['Grocery', 'Outros'], p=dist_is_grocery)
                is_grocery = 1 if is_grocery_cat == 'Grocery' else 0

                Traffic = np.random.choice(['Low', 'Medium', 'High', 'Jam'], p=dist_trafego)
                Weather = np.random.choice(['Stormy', 'Sunny', 'Foggy', 'Windy', 'Sandstorms'], p=dist_clima)
                
                tempo_coleta = np.random.normal(15.0, 5.0)
                if tempo_coleta < 0: tempo_coleta = 0
                
                weekend = np.random.choice([False, True], p=dist_weekend)
                
                Vehicle = np.random.choice(['Moto', 'Scooter', 'Van'], p=dist_veiculo)
                rating_bruto = np.random.normal(media_rating, desvio_padrao_rating)
                Agent_Rating = np.clip(rating_bruto, 3.5, 5.0)
                
                Area = np.random.choice(categorias_area, p=dist_area)
                jam_or_high_traffic = 1 if Traffic in ['High', 'Jam'] else 0
                is_sunny_weather = 1 if Weather == 'Sunny' else 0

                # Dicionário com todos os dados gerados
                dados_pedido = {
                    'Agent_Rating': Agent_Rating, 'Weather': Weather, 'Traffic': Traffic,
                    'Vehicle': Vehicle, 'Area': Area, 'weekend': weekend,
                    'delivery_distance': delivery_distance, 'is_grocery': is_grocery,
                    'jam_or_high_traffic': jam_or_high_traffic,
                    'is_sunny_weather': is_sunny_weather
                }
                
                # 2. Faz a previsão do modelo ANTES de criar o pedido
                previsao = previsao_do_modelo(dados_pedido)

                # 3. Cria a instância do Pedido passando todos os dados e a previsão
                yield Pedido(dados_pedido, previsao)

            self.hold(24 * 60)

class Pedido(sim.Component):
    def __init__(self, dados, previsao):
        super().__init__()
        self.dados = dados
        self.previsao = previsao

    def process(self):

        # 3. Solicita um entregador disponível
        yield self.request(entregadores)
        
        # 5. Calcula o tempo de entrega real
        fator_real = fator_clima[self.dados['Weather']] * fator_trafego[self.dados['Traffic']] * fator_veiculo[self.dados['Vehicle']]
        
        if self.dados['Agent_Rating'] >= 4.5:
            tempo_entrega = (self.dados['delivery_distance'] * 5) * fator_real * fator_rating_impacto['Excelente']
        elif self.dados['Agent_Rating'] >= 4.0:
            tempo_entrega = (self.dados['delivery_distance'] * 5) * fator_real * fator_rating_impacto['Bom']
        else:
            tempo_entrega = (self.dados['delivery_distance'] * 5) * fator_real * fator_rating_impacto['Regular']
            
        self.tempo_real = tempo_entrega
        self.tempo_corrigido = self.tempo_real

        # 6. Insere a Regra Lógica para a Intervenção
        if self.previsao > 120 and self.dados['is_grocery'] == 0 and self.dados['Traffic'] in ['High', 'Jam']:
            self.tempo_corrigido = self.tempo_real - 10

        # 7. Persiste os resultados
        resultados_simulacao.append({
            'id_pedido': self.name(),
            'previsao_min': self.previsao,
            'tempo_real_min': self.tempo_real,
            'tempo_corrigido_min': self.tempo_corrigido,
            'delivery_distance': self.dados['delivery_distance'],
            'Agent_Rating': self.dados['Agent_Rating'],
            'Vehicle': self.dados['Vehicle'],
            'Weather': self.dados['Weather'],
            'Traffic': self.dados['Traffic'],
            'Area': self.dados['Area'],
            'weekend': self.dados['weekend'],
            'is_grocery': self.dados['is_grocery'],
            'jam_or_high_traffic': self.dados['jam_or_high_traffic'],
            'is_sunny_weather': self.dados['is_sunny_weather']
        })

        # 8. Simula o tempo de entrega e libera o entregador
        self.hold(self.tempo_corrigido)
        self.release(entregadores)
        
        return

# --- Configura e roda a simulação ---
print("Simulação iniciada. Aguarde...")

env = sim.Environment(trace=False, yieldless=False)
entregadores = sim.Resource(name='Entregador', capacity=100)
GeradorDePedidos()

env.run(till=7 * 24 * 60)

df_resultados = pd.DataFrame(resultados_simulacao)
print("\n--- Resultados Finais da Simulação ---")
print(df_resultados.head())
print("\n... e mais...")
df_resultados.to_csv(f'../data/simulation/sim_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', sep=',')