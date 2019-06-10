#%%

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler


uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

dados = pd.read_csv(uri)

dicionario = {
    "unfinished": "nao_finalizado",
    "expected_hours": "horas_esperadas",
    "price": "preco"
}

dados = dados.rename(columns = dicionario)

troca = {
    0: 1,
    1: 0
}

dados["finalizado"] = dados["nao_finalizado"].map(troca)

# Separando os dados entre x e y => f(x) = y
dados_x = dados[["horas_esperadas", "preco"]]
dados_y = dados["finalizado"]

# Separando os dados de treino e testes
# Separando os dados de treino e testes
SEED = 5
np.random.seed(SEED)
treino_x_original, teste_x_original, treino_y, teste_y = train_test_split(dados_x, dados_y, test_size = 0.25, stratify = dados_y)

# Usando o StandardScaler para manter a mesma escala entre x e y
scaler = StandardScaler()
scaler.fit(treino_x_original)
treino_x = scaler.transform(treino_x_original)
teste_x = scaler.transform(teste_x_original)

# Treinando o modelo
modelo = SVC()
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acuracia do algoritmo foi {:.2f}% ".format(acuracia))


#%%
