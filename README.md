# Aprendizagem Computacional: Classificação de Spam

## Resumo do Projeto
Este projeto tem como objetivo classificar emails como **Spam** ou **Não Spam** recorrendo a diferentes algoritmos de *Machine Learning*. 

Como o conjunto de dados original apresenta um forte desbalanceamento entre as duas classes, o projeto explora e compara três abordagens diferentes de tratamento de dados (dados originais, SMOTE e Undersampling) para garantir que os modelos aprendem de forma correta e não ficam enviesados.

---

## Modelos e Avaliação
Para esta tarefa de classificação, foram implementados e comparados quatro modelos de classificação:
* **Decision Tree** (Árvore de Decisão)
* **K-Nearest Neighbors** (KNN)
* **Logistic Regression** (Regressão Logística)
* **Naive Bayes**

A avaliação do desempenho de cada modelo baseia-se em métricas críticas para problemas de classificação de spam: **Accuracy** (Exatidão), **Precision** (Precisão) e **Recall** (Revocação).

---

## Balanceamento de Dados
Uma vez que o dataset original não se encontra bem distribuído (existe uma grande discrepância entre a quantidade de emails normais e spam), o projeto está dividido em três abordagens distintas de treino:
1.  **Sem Balanceamento:** Treino com a distribuição original dos dados.
2.  **SMOTE (Oversampling):** Geração de dados sintéticos para a classe minoritária, balanceando as classes no conjunto de treino.
3.  **Undersampling:** Redução aleatória dos dados da classe maioritária para igualar a quantidade da classe minoritária.

---

## Estrutura de Ficheiros

### Código-Fonte (`.py`)
* **`principal.py`:** Executa o fluxo de trabalho com os dados originais, sem aplicar qualquer tipo de balanceamento.
* **`smoted.py`:** Executa o fluxo de trabalho balanceando os dados de treino através da técnica SMOTE.
* **`underSampling.py`:** Executa o fluxo de trabalho balanceando os dados de treino através da técnica de Undersampling.

### Registos de Avaliação (`.csv`)
* Ficheiros gerados automaticamente (como `knn_metrics.csv` e `tree_metrics.csv`) que guardam as métricas de teste obtidas. 
* Estes ficheiros registam o comportamento ao variar os parâmetros dos modelos KNN e Decision Tree para cada abordagem de balanceamento. *Nota: Não é necessário apagar o conteúdo destes ficheiros a cada nova execução, eles atualizam-se com os novos testes.*

---

## Como Executar

1.  **Pré-requisitos:** Certifique-se de que tem o Python e as bibliotecas necessárias instaladas. Pode instalar as dependências correndo:
    ```bash
    pip install numpy pandas scikit-learn seaborn matplotlib imbalanced-learn
    ```
    *(Nota: as bibliotecas `time` e `collections` são nativas do Python).*

2.  **Execução:**
    Basta correr o script Python correspondente à abordagem que deseja testar. Por exemplo, no seu terminal, execute:
    ```bash
    python principal.py
    ```

3.  **Resultados:** À medida que o código corre, as métricas de cada modelo serão exibidas automaticamente no terminal, acompanhadas pela abertura de janelas com os respetivos gráficos e pela atualização dos ficheiros `.csv` locais.
