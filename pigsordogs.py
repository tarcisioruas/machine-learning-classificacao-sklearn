from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Features (1 - Sim, 2 - Não)
# Tem pelo longo?
# Tem perna curta?
# faz au au

# Definindo porcos
porco1 = [0, 1, 0] # Não tem pelo longo, tem perna curta, não faz au au
porco2 = [0, 1, 1] # Nao tem pelo longo, tem perna curta e "faz au au"
porco3 = [1, 1, 0] # Tem pelo longo, tem perna curta, não faz au au

# Definindo cachorros
cachorro1 = [0, 1, 1] # Não tem pelo longo, tem perna curta, faz au au
cachorro2 = [1, 0, 1] # Tem pelo longo, não tem perna curta e faz au au
cachorro3 = [1, 1, 1] # Tem pelo longo, tem perna curta, faz au au

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 - porco e 0 - cachorro
treino_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
# Treinando o modelo
model.fit(treino_x, treino_y)

# Testando modelo
animal1 = [1, 1, 1] #cachorro
animal2 = [1, 1, 0] #porco
animal3 = [0, 1, 1] #cachorro

teste_x = [animal1, animal2, animal3]
teste_y = [0, 1, 1]

previsoes = model.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("Taxa de acerto = {:.2f}%".format(acuracia)) 