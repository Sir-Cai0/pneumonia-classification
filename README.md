# Classificação de Pneumonia em Raios-X com Deep Learning

Este projeto é uma demonstração prática do desenvolvimento ponta a ponta de um **Pipeline de Machine Learning** aplicado à Visão Computacional. O objetivo do modelo é analisar imagens de raios-X do tórax e classificar os pacientes em duas categorias: **NORMAL** ou **PNEUMONIA**.

O foco principal do projeto foi estruturar um fluxo de trabalho eficiente de Deep Learning, passando pela extração e processamento de dados, construção da arquitetura, treinamento acelerado com Transfer Learning e avaliação crítica das métricas preditivas.

## Arquitetura e Pipeline Técnico

O projeto foi inteiramente desenvolvido em Python utilizando o framework **TensorFlow/Keras** no ambiente do Google Colab. O pipeline consiste nas seguintes etapas:

1. **Ingestão de Dados:** Download via API e carregamento otimizado de imagens utilizando `image_dataset_from_directory`.
2. **Otimização de Performance:** Aplicação de `.prefetch()` com `tf.data.AUTOTUNE` para otimizar o uso de CPU/GPU e evitar gargalos de memória I/O durante o treinamento.
3. **Transfer Learning (EfficientNetV2S):** Utilização da arquitetura estado-da-arte **EfficientNetV2Small**, pré-treinada na ImageNet. A rede base teve seus pesos congelados, e novas camadas densas (*Dense*, *BatchNormalization*, *Dropout*) foram adicionadas no topo para adaptação ao problema de classificação binária médica.
4. **Treinamento e Regularização:** O modelo foi compilado com otimizador `Adam` e função de perda `categorical_crossentropy`. As camadas de regularização nativas da rede garantiram um aprendizado robusto sem *overfitting*.

## Resultados e Avaliação do Modelo

### Gráficos de Treino e Validação:
O modelo apresentou resultados excepcionais logo nas primeiras épocas de treinamento, alcançando uma **Acurácia de Validação de aproximadamente 97,4%**.

<img width="1319" height="549" alt="image" src="https://github.com/user-attachments/assets/85ad2c1c-8054-4f38-aa4e-19cd28401d5d" />

### Matriz de Confusão e Impacto no Diagnóstico:
Para além da acurácia global, o desempenho do modelo foi validado através de uma Matriz de Confusão, avaliando o custo real de cada tipo de erro:

<img width="652" height="551" alt="image" src="https://github.com/user-attachments/assets/a7aaf5d3-8824-459d-bf99-d0a8201ff289" />

**Insights Clínicos do Modelo:**
* **Baixíssimo índice de Falsos Negativos (11 casos):** Num contexto médico, o pior cenário é classificar um paciente doente como saudável. O modelo minimizou criticamente este erro, o que o torna altamente seguro.
* **Falsos Positivos sob controle (16 casos):** Apenas uma pequena fração de exames normais precisaria de uma segunda avaliação médica, otimizando o fluxo de triagem sem colocar vidas em risco.

## 📂 Dataset:
Os dados utilizados são públicos e foram extraídos do Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). 
*As imagens foram redimensionadas para (224, 224, 3) para compatibilidade com a arquitetura base.*

## Como reproduzir o projeto:

O pipeline completo está documentado de forma clara e linear. Para executá-lo:
1. Faça o clone deste repositório.
2. Abra o arquivo `.ipynb` no **Google Colab**.
3. Insira suas credenciais (Token API) do Kaggle na primeira célula para o download automático do dataset.
4. Ative a aceleração por Hardware (T4 GPU) no Colab.
5. Execute todas as células sequencialmente.

---
**Tecnologias Utilizadas:** `Python`, `TensorFlow`, `Keras`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`.
