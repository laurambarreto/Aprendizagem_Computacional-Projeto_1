1. Descrição
Este projeto tem como objetivo classificar emails como spam ou não spam e utiliza diferentes modelos de Machine 
Learning para tal, nomeadamente: Decision Tree, KNN, Regressão Logística, e Naive Bayes. 
Para avaliar estes modelos, usámos métricas como: accuracy, precision, e recall.
Nota: Visto que o dataset não está bem distribuído, usámos duas abordagens de balanceamento: SMOTE e underSampling.

2. principal.py
Trabalho com os dados sem qualquer balanceamento efetuado. 

3. smoted.py
Trabalho com os dados de treino balanceados através do SMOTE.

4. underSampling.py
Trabalho com os dados de treino balanceados através de underSampling.

5. ficheiros .csv
Usados para guardar as métricas de teste, obtidas ao usar vários parâmetros nos modelos knn e decision Tree, para 
cada abordagem de balanceamento. Ou seja, quando corrido o ficheiro principal.py, vai testar vários parâmetros
para estes dois modelos e guardar as métricas do modelo KNN no ficheiro knn_metrics.csv e do modelo decision Tree
no ficheiro tree_metrics.csv, de igual modo acontece para os ficheiros smoted.py e underSampling.py.
Nota: Não é necessário apagar o conteúdo destes ficheiros em cada execução.

6. Como executar
Basta correr os ficheiros python que desejar e as métricas de cada modelo serão exibidas automaticamente no 
terminal à medida que os gráficos e resultados forem surgindo.

7. Biliotecas utilizadas
numpy
pandas
scikit-learn
seaborn
matplotlib
imblearn
time
collections