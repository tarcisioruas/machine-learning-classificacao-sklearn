import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

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
SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(dados_x, dados_y, test_size = 0.25, random_state = SEED, stratify = dados_y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# Treinando o modelo
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

# Testando o modelo
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100

print("Acuracia desse algoritmo aplicado a esses dados é igual à {:.2f}%".format(acuracia))

# A acuracia acima é boa? Vamos verificar
previsoes_inventadas = np.ones(540)
acuracia_inventada = accuracy_score(teste_y, previsoes_inventadas) * 100
print("Acuracia baseada em chutes {:.2f}%".format(acuracia_inventada))