# Import das bibliotecas necessárias
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
from sklearn.tree import plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('spambase.csv', delimiter = ",")

# Seleção das colunas das características
X = df.drop("spam", axis=1)

# Seleção da coluna target
y = df.spam

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42) 

# Função que retorna as métricas de avaliação
def metricas(y_true, y_pred):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)

##---------- Análise inicial ----------##
# Informações sobre o Dataset
print(df.info(), "\n")

# Distribuição de spam e não spam
sns.countplot(x = 'spam',data = df)
plt.title("Spam distribution")
plt.show()

print(df['spam'].value_counts(), "\n")

# Correlações entre todas as colunas
correlation_matrix = df.corr()
plt.figure(figsize = (6, 4))
sns.heatmap(correlation_matrix,cmap = 'coolwarm', annot = False)
plt.title('Correlation Matrix Heatmap')
plt.show()

##---------- Pré-processamento ----------##
# Normalizar os dados (importante para KNN e regressão logística)
scaler = StandardScaler()
X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_test_norm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

##----------- CLUSTERING ---------##
# Normalizar os dados
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# Reduzir a dimensionalidade para 2D para visualização (PCA)
pca = PCA(n_components = 2)
X_reduced = pca.fit_transform(X_norm)

# Aplicar K-Means
kmeans = KMeans(n_clusters = 2, random_state = 42, n_init = 10) # init, escolhe apenas o melhor resultado

# Prever os clusters
clusters = kmeans.fit_predict(X_reduced)

# Visualizar os clusters comparando com as classes reais
plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = clusters, cmap = 'viridis', alpha = 0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'red', marker = 'x', label = "Centroids")
plt.title("Clustering com k-means")
plt.legend()
plt.subplot(1,2,2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y, cmap = 'coolwarm', alpha = 0.5)
plt.title("Classes Reais (Spam vs. Não Spam)")
plt.show()

##---------- ÁRVORE DE DECISÃO ----------##
# Experiências com diferentes profundidades
results=[]
for depth in range(1,11):
    # Criar o modelo de árvore de decisão
    clf = DecisionTreeClassifier (criterion = 'entropy', max_depth = depth, random_state = 42)

    # Treinar o modelo com os dados de treino
    clf.fit(X_train, y_train)

    # Prever resultados através da árvore de decisão
    y_train_pred = clf.predict (X_train)
    y_test_pred = clf.predict (X_test)

    # Calcular métricas para os dados de teste
    accuracy, recall, precision = metricas(y_test, y_test_pred)
    
    # Guardar os resultados
    results.append([depth, accuracy, recall, precision])

# Criar um DataFrame para a Árvore de Decisão
dt_results_df = pd.DataFrame(results, columns = ['Depth', 'Accuracy', 'Recall', 'Precision'])

# Guardar os resultados num ficheiro CSV
dt_results_df.to_csv('tree_metrics.csv', index = False, mode = 'w')
print("Métricas da Árvore de Decisão guardadas em 'tree_metrics.csv'", "\n")

# Plot das métricas da Árvore de Decisão com diferentes profundidades
plt.figure(figsize = (10, 6))
plt.plot(dt_results_df['Depth'], dt_results_df['Accuracy'], marker = 'o', label = 'Accuracy')
plt.plot(dt_results_df['Depth'], dt_results_df['Recall'], marker = 's', label = 'Recall')
plt.plot(dt_results_df['Depth'], dt_results_df['Precision'], marker = '^', label = 'Precision')

# Garantir que todas as profundidades aparecem no eixo X
plt.xticks(dt_results_df['Depth'])

# Definir limites do eixo Y de 0.5 a 1
plt.ylim(0.5, 1)
plt.xlabel('Profundidade da Árvore (Depth)' )
plt.ylabel('Valor da Métrica')
plt.title('Métricas com Diferentes Profundidades (Árvore)', fontsize = 18)
plt.legend(fontsize = 15)
plt.grid()
plt.show()

# Criar modelo de árvore de decisão com profundidade 4
clf = DecisionTreeClassifier (criterion = 'entropy', max_depth = 4, random_state = 42)

# Medir o tempo de treino
start_time = time.time()

# Treinar o modelo com os dados de treino
clf.fit (X_train, y_train)
training_time = time.time() - start_time

# Prever resultados através da árvore de decisão
y_train_pred = clf.predict (X_train)
y_test_pred = clf.predict (X_test)

# Visualizar a árvore de decisão
plt.figure (1, figsize = (250,15))
plot_tree (clf, filled = True, feature_names = X.columns, class_names = ['not spam','spam'], fontsize = 6)
plt.title("Decision Tree")
plt.show ()

# Analisar as diferentes métricas
accuracy_train, recall_train, precision_train = metricas(y_train, y_train_pred)
accuracy_test, recall_test, precision_test = metricas(y_test, y_test_pred)

print("Decision Tree:")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, '\n')

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot =True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with Decision Tree', fontsize = 18)
# Ajustar o tamanho dos nomes dos eixos x e y
ax.set_xlabel('Predicted Values', fontsize = 14)
ax.set_ylabel('Actual Values', fontsize = 14)
# Ajustar o tamanho dos valores dentro da matriz de confusão
for text in ax.texts:
    text.set_fontsize(16)
plt.show()

##---------- KNN ----------##
# Experiências com vários k
results = []
for k in range(1,16):
    knn = KNN (n_neighbors = k) 

    # Treinar o modelo KNN
    knn.fit(X_train, y_train)
   
    # Prever os resultados com o modelo KNN
    y_train_pred = knn.predict (X_train)
    y_test_pred = knn.predict (X_test)

     # Calcular métricas para os dados de teste
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    
    # Guardar os resultados
    results.append([k, accuracy, recall, precision])

# Criar um DataFrame
results_df = pd.DataFrame(results, columns = ['k', 'Accuracy', 'Recall', 'Precision'])

# Abrir o ficheiro em modo de escrita para apagar o conteúdo antes de escrever
open('knn_metrics.csv', 'w').close()

# Escrever as métricas no ficheiro
results_df.to_csv('knn_metrics.csv', index = False)
print("Métricas do knn guardadas em 'knn_metrics.csv'", "\n")

# Plot das métricas com diferentes valores de k
plt.figure(figsize = (10, 6))
plt.plot(results_df['k'], results_df['Accuracy'], marker = 'o', label = 'Accuracy')
plt.plot(results_df['k'], results_df['Recall'], marker = 's', label = 'Recall')
plt.plot(results_df['k'], results_df['Precision'], marker = '^', label = 'Precision')
# Definir limites do eixo Y de 0.5 a 1
plt.ylim(0.5, 1)
# Garantir que todas as profundidades aparecem no eixo X
plt.xticks(results_df['k'])
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Valor da Métrica')
plt.title('Métricas com Diferentes Valores de k', fontsize = 18)
plt.legend(fontsize = 15)
plt.grid()
plt.show()

# Usar k = 1 que obteve as melhores resultados
knn = KNN (n_neighbors = 1)

# Medir o tempo de treino
start_time = time.time()
# Treinar o modelo com os dados de treino
knn.fit (X_train, y_train)
training_time = time.time() - start_time

# Prever os resultados com o modelo KNN
y_train_pred = knn.predict (X_train)
y_test_pred = knn.predict (X_test)

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot = True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with KNN', fontsize = 18)
ax.set_xlabel('\nPredicted Spam ', fontsize = 14)
ax.set_ylabel('Actual Spam ', fontsize = 14)
for text in ax.texts:
    text.set_fontsize(16)
plt.show()

# Analisar as métricas 
accuracy_train, recall_train, precision_train = metricas(y_train, y_train_pred)
accuracy_test, recall_test, precision_test = metricas(y_test, y_test_pred)
print("KNN:")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, '\n')

##---------- REGRESSÃO LOGÍSTICA ----------##
# Criar o modelo de regressão logística
model = LogisticRegression()

# Medir o tempo de treino
start_time = time.time()
# Treinar o modelo
model.fit(X_train_norm, y_train)
training_time = time.time() - start_time

# Fazer previsões com os dados de teste normalizados
y_test_pred = model.predict(X_test_norm)
y_train_pred = model.predict(X_train_norm)

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot = True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with Logistic Regression', fontsize = 18)
ax.set_xlabel('\nPredicted Spam Category', fontsize = 14)
ax.set_ylabel('Actual Spam Category ', fontsize = 14)
for text in ax.texts:
    text.set_fontsize(16)
plt.show()

# Analisar as métricas 
accuracy_train, recall_train, precision_train = metricas(y_train, y_train_pred)
accuracy_test, recall_test, precision_test = metricas(y_test, y_test_pred)
print("Logistic Regression:")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, '\n')

##---------- NAIVE BAYES ----------##
# Inicializar o modelo
gnb = GaussianNB()

# Medir o tempo de treino
start_time = time.time()
# Treinar o modelo
gnb.fit(X_train, y_train)
training_time = time.time() - start_time

# Prever o resultado com gaussian
y_test_pred = gnb.predict(X_test)
y_train_pred = gnb.predict(X_train)

# Matriz de confusão para Gaussian
cm = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(cm, annot = True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with Naive Bayes', fontsize = 18)
ax.set_xlabel('\nPredicted Values', fontsize = 14)
ax.set_ylabel('Actual Values ', fontsize = 14)
for text in ax.texts:
    text.set_fontsize(16)
plt.show()

# Analisar as métricas
accuracy_train, recall_train, precision_train = metricas(y_train, y_train_pred)
accuracy_test, recall_test, precision_test = metricas(y_test, y_test_pred)

print("Naive Bayes:")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, '\n')