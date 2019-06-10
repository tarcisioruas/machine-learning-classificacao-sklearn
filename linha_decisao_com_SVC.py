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

# Quando houve a reescalação, as colunas horas_esperadas e preco se perderam,
# mas ainda podem encontradas usando indices
x_min = teste_x[:,0].min()
x_max = teste_x[:,0].max()
y_min = teste_x[:,1].min()
y_max = teste_x[:,1].max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x[:,0], teste_x[:,1], c=teste_y, s=1)

#%%
