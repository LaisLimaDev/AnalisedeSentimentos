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


