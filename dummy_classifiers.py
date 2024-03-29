
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
dados = pd.read_csv(uri)

dicionario = { 
    'mileage_per_year': 'milhas_por_ano',
    'model_year': 'ano_do_modelo',
    'price': 'preco',
    'sold':'vendido'
}

trocar_resultados = {
    'yes': 1,
    'no': 0
}

dados = dados.rename(columns = dicionario)

# Transformando as colunas
dados.vendido = dados.vendido.map(trocar_resultados)
dados['idade_do_modelo'] = datetime.today().year - dados.ano_do_modelo
dados['km_por_ano'] = dados.milhas_por_ano * 1.60934

# Apagando as colunas desnecessárias
dados = dados.drop(columns = ["Unnamed: 0", "milhas_por_ano", "ano_do_modelo"], axis = 1)

# Separando os dados dos resultados
dados_x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
dados_y = dados["vendido"]

SEED = 5
np.random.seed(SEED)

treino_x_original, teste_x_original, treino_y, teste_y = train_test_split(dados_x, dados_y, test_size = 0.25, stratify = dados_y)

# Dummy 
dummy_stratified = DummyClassifier()
dummy_stratified.fit(treino_x_original, treino_y)
previsoes = dummy_stratified.predict(teste_x_original)
acuracia = dummy_stratified.score(teste_y, previsoes) * 100
print("A acurácia do dummy_stratified é {:.2f}%".format(acuracia))

# LinearSVC 
model = LinearSVC()
model.fit(treino_x_original, treino_y)
previsoes = model.predict(teste_x_original)
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia do dummy_stratified é {:.2f}%".format(acuracia))


# SVC
scaler = StandardScaler()
scaler.fit(treino_x_original)
treino_x = scaler.transform(treino_x_original)
teste_x = scaler.transform(teste_x_original)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia do SVC foi {:.2f}%".format(acuracia))