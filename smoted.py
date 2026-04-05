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
from imblearn.over_sampling import SMOTE
from collections import Counter

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('spambase.csv', delimiter = ",")

# Seleção das colunas das características
X = df.drop("spam", axis=1)

# Seleção da coluna target
y = df.spam

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42) 

# Função que retorna as métricas de avaliação
def metricas(y_pred, y_true):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)

##---------- Anélise inicial ----------##
# Informações sobre o Dataset
print(df.info(), "\n")

# Distribuição de spam e não spam
sns.countplot(x = 'spam',data = df)
plt.title("Spam distribution", fontsize = 22)
# Aumentar o tamanho das palavras dos rótulos dos eixos
plt.xlabel('Spam', fontsize = 18)  
plt.ylabel('Count', fontsize = 18)
# Aumentar o tamanho das palavras nos eixos
plt.tick_params(axis = 'x', labelsize = 18)  
plt.tick_params(axis = 'y', labelsize = 18)
plt.show()

# Correlações entre todas as colunas
correlation_matrix = df.corr()
plt.figure(figsize = (6, 4))
sns.heatmap(correlation_matrix,cmap = 'coolwarm', annot = False)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Distribuição de spam e não spam nos dados de treino antes do SMOTE
sns.countplot(x = y_train)
plt.title("Spam distribution (train)", fontsize = 22)
# Aumentar o tamanho das palavras dos rótulos dos eixos
plt.xlabel('Spam', fontsize = 18)  
plt.ylabel('Count', fontsize = 18)
# Aumentar o tamanho das palavras nos eixos
plt.tick_params(axis = 'x', labelsize = 18)  
plt.tick_params(axis = 'y', labelsize = 18)
plt.show()

##---------- Pré-processamento ----------##
# Aplicar SMOTE aos dados de treino
smote = SMOTE(sampling_strategy = 'auto', random_state = 42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

# Distribuição de spam e não spam com SMOTE
sns.countplot(x = y_train_SMOTE)
plt.title("Spam distribution (train with SMOTE)", fontsize = 22)
# Aumentar o tamanho das palavras dos rótulos dos eixos
plt.xlabel('Spam', fontsize = 18)  
plt.ylabel('Count', fontsize = 18)
# Aumentar o tamanho das palavras nos eixos
plt.tick_params(axis = 'x', labelsize = 18)  
plt.tick_params(axis = 'y', labelsize = 18)
plt.show()

# Normalizar os dados (importante para regressão logística e KNN)
scaler = StandardScaler()
X_train_SMOTE_norm = pd.DataFrame(scaler.fit_transform(X_train_SMOTE), columns = X_train_SMOTE.columns)
X_test_norm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

##---------- ÁRVORE DE DECISÃO ----------##
results = []
for depth in range(1,11):
    # Criar o modelo de árvore de decisão
    clf = DecisionTreeClassifier (criterion = 'entropy', max_depth = depth, random_state = 42)

    # Treinar o modelo com os dados de treino
    clf.fit (X_train_SMOTE, y_train_SMOTE)

    # Prever resultados através da árvore de decisão
    y_train_pred = clf.predict (X_train_SMOTE)
    y_test_pred = clf.predict (X_test)

    # Calcular métricas para os dados de teste
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    
    # Guardar os resultados
    results.append([depth, accuracy, recall, precision])

# Criar um DataFrame para a Árvore de Decisão
dt_results_df = pd.DataFrame(results, columns = ['Depth', 'Accuracy', 'Recall', 'Precision'])

# Guardar os resultados num ficheiro CSV
dt_results_df.to_csv('tree_metrics_SMOTE.csv', index = False, mode = 'w')

print("Métricas da Árvore de Decisão guardadas em 'tree_metrics_SMOTE.csv'", '\n')

# Plot das métricas da Árvore de Decisão
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
plt.title('Métricas com Diferentes Profundidades (Árvore de Decisão-SMOTE)', fontsize=18)
plt.legend(fontsize = 15)
plt.grid()
plt.show()

# Inicializar o modelo com profundidade 4
clf = DecisionTreeClassifier (criterion = 'entropy', max_depth = 4,random_state = 42)

# Medir o tempo de treino
start_time = time.time()
# Treinar o modelo com os dados de treino
clf.fit (X_train_SMOTE, y_train_SMOTE)
training_time = time.time() - start_time

# Prever resultados através da árvore de decisão
y_train_pred = clf.predict (X_train_SMOTE)
y_test_pred = clf.predict (X_test)

# Visualizar a árvore de decisão 
plot_tree (clf, feature_names = X.columns, filled = True, class_names = ['not spam','spam'], fontsize = 5)
plt.title("Decision tree with SMOTE", fontsize=18)
plt.show ()

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot = True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with Decision Tree (SMOTE)', fontsize = 18)
ax.set_xlabel('\nPredicted Values', fontsize = 14)
ax.set_ylabel('Actual Values ', fontsize = 14)
plt.show()

# Analisar as diferentes métricas
accuracy_train, recall_train, precision_train = metricas(y_train_pred, y_train_SMOTE)
accuracy_test, recall_test, precision_test = metricas(y_test_pred, y_test)
print("Decision Tree (SMOTE):")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, '\n')

##---------- KNN ----------## 
# Experiências com vários k
results = []
for k in range(1,16):
    knn = KNN (n_neighbors = k) 
   
    # Treinar o modelo com os dados de treino
    knn.fit (X_train_SMOTE_norm, y_train_SMOTE)

    # Prever os resultados com o modelo KNN
    y_train_pred = knn.predict (X_train_SMOTE_norm)
    y_test_pred = knn.predict (X_test_norm)

    # Calcular métricas para os dados de teste
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    
    # Guardar os resultados
    results.append([k, accuracy, recall, precision])

# Criar um DataFrame
results_df = pd.DataFrame(results, columns = ['k', 'Accuracy', 'Recall', 'Precision'])

# Abrir o ficheiro em modo de escrita para apagar o conteúdo antes de escrever
open('knn_metrics_SMOTE.csv', 'w').close()

# Escrever as métricas no ficheiro
results_df.to_csv('knn_metrics_SMOTE.csv', index = False)
    
print("Métricas do KNN guardadas em 'knn_metrics_SMOTE.csv'", '\n')

# Plot das métricas
plt.figure(figsize = (10, 6))
plt.plot(results_df['k'], results_df['Accuracy'], marker = 'd', label = 'Accuracy')
plt.plot(results_df['k'], results_df['Recall'], marker = 's', label = 'Recall')
plt.plot(results_df['k'], results_df['Precision'], marker = '^', label = 'Precision')
# Definir limites do eixo Y de 0.5 a 1
plt.ylim(0.5, 1)
# Garantir que todas os k aparecem no eixo X
plt.xticks(results_df['k'])
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Valor da Métrica')
plt.title('Métricas com Diferentes Valores de k (SMOTE)', fontsize = 18)
plt.legend(fontsize = 15)
plt.grid()
plt.show()

# Usar k = 1 que obteve as melhores métricas
knn = KNN (n_neighbors = 11) 

# Medir o tempo de treino 
start_time = time.time()
# Treinar o modelo com os dados de treino
knn.fit (X_train_SMOTE_norm, y_train_SMOTE)
training_time = time.time() - start_time    

# Prever os resultados com o modelo KNN
y_train_pred = knn.predict (X_train_SMOTE_norm)
y_test_pred = knn.predict (X_test_norm)

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot = True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with KNN (SMOTE)', fontsize = 20)
ax.set_xlabel('\nPredicted Values', fontsize = 14)
ax.set_ylabel('Actual Values ', fontsize = 14)
for text in ax.texts:
    text.set_fontsize(16)
plt.show()

# Analisar as métricas
accuracy_train, recall_train, precision_train = metricas(y_train_pred, y_train_SMOTE)
accuracy_test, recall_test, precision_test = metricas(y_test_pred, y_test) 
print("KNN (SMOTE):")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, '\n')

##---------- REGRESSÃO LOGÍSTICA ----------##
# Inicializar o modelo
model = LogisticRegression()

# Medir o tempo de treino
start_time = time.time()
# Treinar o modelo
model.fit(X_train_SMOTE_norm, y_train_SMOTE)
training_time = time.time() - start_time

# Fazer previsões com os dados de teste
y_test_pred = model.predict(X_test_norm)
y_train_pred = model.predict(X_train_SMOTE_norm)

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot = True, cmap = 'Blues',fmt = "g", annot_kws = {"size": 18})
ax.set_title('Confusion Matrix with LR (SMOTE)', fontsize = 20)
ax.set_xlabel('\nPredicted Values', fontsize = 14)
ax.set_ylabel('Actual Values ', fontsize = 14)
plt.show()

# Analisar as métricas 
accuracy_train, recall_train, precision_train = metricas(y_train_pred, y_train_SMOTE)   
accuracy_test, recall_test, precision_test = metricas(y_test_pred, y_test)
print("Logistic Regression (SMOTE):")
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
gnb.fit(X_train_SMOTE, y_train_SMOTE)
training_time = time.time() - start_time    

# Prever os resultados
y_test_pred = gnb.predict(X_test)
y_train_pred = gnb.predict(X_train_SMOTE)

# Matriz de confusão para Naive Bayes
cm = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(cm, annot = True, cmap = 'Blues',fmt = "g")
ax.set_title('Confusion Matrix with Naive Bayes (SMOTE)', fontsize = 20)
ax.set_xlabel('\nPredicted Values', fontsize = 14)
ax.set_ylabel('Actual Values ', fontsize = 14)
plt.show()

# Analisar as métricas 
accuracy_train, recall_train, precision_train = metricas(y_train_pred, y_train_SMOTE)   
accuracy_test, recall_test, precision_test = metricas(y_test_pred, y_test)
print("Naive Bayes (SMOTE):")
print("Training time: ", training_time)
print ("Train accuracy: " , accuracy_train)
print ("Train recall: " , recall_train)
print ("Train precision: " , precision_train)
print ("Test accuracy: ", accuracy_test)
print ("Test recall: ", recall_test)
print ("Test precision: ", precision_test, "\n")