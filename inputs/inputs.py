# Bibliotecas de tokenização e Teorema de Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Frases iniciais

frases = [
    "Produto excelente",
    "Ótimo custo-benefício",
    "Demorou demais para chegar!!",
    "Chegou quebrado por que o entregador jogou pelo portão",
    "Agora só compro produto dessa marca"
    
]

# 1 = positivo 0 = negativo

rotulos = [1, 1, 0, 0, 1]

vetorizador = CountVectorizer()
x = vetorizador.fit_transform(frases)

# Criação do modelo

modelo = MultinomialNB()
modelo.fit (x, rotulos)


novas_frases = [
    "Adorei, superou todas as minhas expectativas",
    "O produto é horrível",
    "Muito bom!!"
]

X_teste = vetorizador.transform(novas_frases)
previsoes = modelo.predict(X_teste)

for i, frase in enumerate(novas_frases):
    if previsoes[i] == 1:
        print(frase, "é uma frase positiva")
    else:
        print(frase, "é uma frase negativa")
