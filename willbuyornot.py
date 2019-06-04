from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

uri='https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
dados = pd.read_csv(uri)


# Renomeando as colunas
mapa = {
    "home": "principal",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "comprou"
}

SEED = 20

dados = dados.rename(columns = mapa)

# separando as colunas referentes a x e y 
dados_x = dados[["principal", "como_funciona", "contato"]]
dados_y = dados["comprou"]

# separando dados de treino e testes
treino_x, teste_x, treino_y, teste_y = train_test_split(dados_x, dados_y, test_size = 0.25, random_state = SEED, stratify = dados_y)

print("Treinaremos com {} elementos e testaremos com {} elementos.".format(len(treino_y), len(teste_y)))

## iniciando o treinamento
model = LinearSVC()

# Treinando o modelo
model.fit(treino_x, treino_y)

# Iniciando o teste
previsoes = model.predict(teste_x)

# Verificando a acuracia
acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acuracia do algoritmo para o conjunto é de {:.2f}%".format(acuracia))

