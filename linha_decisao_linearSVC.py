#%%

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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
SEED = 5
treino_x, teste_x, treino_y, teste_y = train_test_split(dados_x, dados_y, test_size = 0.25, random_state = SEED, stratify = dados_y)

# Treinando o modelo
modelo = LinearSVC(random_state = 0, max_iter = 1000)
modelo.fit(treino_x, treino_y)

# horas_esperadas no eixo x
# preco no eixo y
x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

# print(x_min, x_max, y_min, y_max)
# Desenharemos um gráfico de 100 x 100 pixels
pixels = 100
pontos_x = np.arange(x_min, x_max, (x_max - x_min)/pixels)
pontos_y = np.arange(y_min, y_max, (y_max - y_min)/pixels)

# Criando o grid com os dados do eixo x e y
xx, yy = np.meshgrid(pontos_x, pontos_y)

# Mesclando os pontos
pontos = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)
print(Z)

# plotando o gráfico com a linha de decisão do algoritmo
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

#%%
