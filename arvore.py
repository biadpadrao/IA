# Baseado no código disponibilizado no Canvas
# Beatriz Demetrio Ribeiro Padrão

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn import tree

# Abrindo o CSV
base = pd.read_csv('restaurante.csv', sep=';')

# np.unique(base['Conc'], return_counts=True)

# Separando atributos de entrada e classe
X_prev = base.iloc[:, 0:10].values
Y_classe = base.iloc[:, 10].values

# Transformando os ordinais em numéricos
def transform_with_labelEncoder(df, col:int):
  df[:, col] = LabelEncoder().fit_transform(df[:, col])

# Base duplicada para mudar os valores 
X_encoded = X_prev.copy()

transform_with_labelEncoder(X_encoded, 0);
transform_with_labelEncoder(X_encoded, 1);
transform_with_labelEncoder(X_encoded, 2);
transform_with_labelEncoder(X_encoded, 3);
transform_with_labelEncoder(X_encoded, 4);
transform_with_labelEncoder(X_encoded, 5);
transform_with_labelEncoder(X_encoded, 6);
transform_with_labelEncoder(X_encoded, 7);
transform_with_labelEncoder(X_encoded, 9);

# OneHotEncoder - binarizando atributos não ordinais
one_hot_encoder_transformer = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')
X_encoded = one_hot_encoder_transformer.fit_transform(X_encoded)

# Separados em TREINO e TESTE
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_encoded, Y_classe, test_size = 0.20, random_state = 23)

# Implementação do algoritmo Decision Tree
modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, Y_treino)

# Teste
previsoes = modelo.predict(X_teste)

# print(confusion_matrix(Y_teste, previsoes))

cm = ConfusionMatrix(modelo)
cm.fit(X_treino, Y_treino)
cm.score(X_teste, Y_teste)

# Imprimindo a matriz de confusão
plt.figure(figsize=(8, 13))
plt.show()

# Gerando da árvore
previsores = ['Frances','Hamburguer','Italiano','Tailandes','Alternativo','Bar','SexSab','Fome','Cliente','Preço','Chuva','Res','Tipo','Tempo']
tree.plot_tree(modelo, feature_names=previsores,class_names=modelo.classes_.tolist(), filled=True)

plt.savefig('arvoreDecisao.png')
