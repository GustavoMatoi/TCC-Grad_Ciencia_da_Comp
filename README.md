# 🧠 Classificação Automatizada da Toxoplasmose Ocular

Este repositório apresenta um **método computacional para a classificação automatizada da toxoplasmose ocular** em imagens de fundo de olho, utilizando **extração de características radiômicas** combinada com **algoritmos de aprendizado de máquina e deep learning**.

A toxoplasmose ocular é uma condição grave que pode causar **lesões retinianas irreversíveis e cegueira**, tornando essencial o desenvolvimento de ferramentas que auxiliem o diagnóstico precoce e preciso.

---

## 📌 Contexto do Problema

A toxoplasmose afeta aproximadamente **33% da população mundial**, sendo causada pelo parasita *Toxoplasma gondii*.  
No Brasil, estima-se que **1 a cada 3 pessoas** seja infectada, com milhares de casos registrados nos últimos anos.

Quando a infecção atinge os olhos, a doença se manifesta como **toxoplasmose ocular**, caracterizada por lesões na retina que podem variar em formato, tamanho e localização, dificultando o diagnóstico clínico baseado apenas na avaliação visual.

---

## 🎯 Objetivo

Desenvolver um **método automático e de baixo custo computacional** para:
- Identificar padrões em imagens de fundo de olho
- Classificar casos saudáveis e com toxoplasmose ocular
- Apoiar oftalmologistas no diagnóstico clínico

---

## 🧪 Base de Dados

- Imagens de fundo de olho de pacientes diagnosticados com toxoplasmose ocular
- Coletadas em dois hospitais no Paraguai:
  - Hospital de Clínicas
  - Hospital General Pediátrico Acosta Ñu
- Complementadas com imagens saudáveis da base **FIRE (Fundus Image Registration Dataset)**

📊 **Base final balanceada**:
- 562 imagens
- 281 imagens saudáveis
- 281 imagens com toxoplasmose ocular

Todas as imagens foram padronizadas para **512×512 pixels**.

---

## 🛠️ Metodologia

### 🔹 Pré-processamento
As imagens passaram pelas seguintes etapas:
1. Redimensionamento
2. Extração do canal verde
3. Aplicação de CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. Geração da imagem negativa

Essas etapas visam reduzir variabilidades de aquisição e destacar lesões retinianas.

---

### 🔹 Extração de Características Radiômicas

Foram extraídas **220 características**, utilizando:

- **PyRadiomics**  
  - Estatísticas de primeira ordem  
  - GLCM, GLRLM, GLSZM, GLDM, NGTDM  

- **Mahotas**  
  - Local Binary Patterns (LBP)  
  - Momentos de Zernike  
  - Threshold Adjacency Statistics (TAS)

A região de interesse (ROI) foi definida como **toda a retina**, devido à variabilidade espacial das lesões.

---

### 🔹 Seleção e Normalização
- Seleção das **100 características mais relevantes** usando **SelectKBest**
- Normalização dos dados com **MinMax Scaling**

---

## 🤖 Algoritmos Utilizados

Os vetores de características foram utilizados para treinar os seguintes modelos:

- Support Vector Machine (SVM)
- Decision Tree (DT)
- Random Forest (RF)
- Stochastic Gradient Descent (SGD)
- AdaBoost
- XGBoost
- Multilayer Perceptron (MLP)
- Fully Connected Neural Network (FCNN)

Os modelos foram implementados majoritariamente em **Python**, utilizando **Scikit-learn** e **Keras**.

---

## 📊 Avaliação Experimental

- Validação cruzada **K-Fold (k = 10)**
- Métricas utilizadas:
  - Acurácia (ACC)
  - F1-Score
  - Precisão
  - Recall
  - Área sob a curva (AUC)

### 🔥 Resultados
- Todos os modelos obtiveram **desempenho superior a 90%**
- Melhor desempenho:
  - **SVM (kernel polinomial)** e **MLP**
  - Até **96% de acurácia**
  - **AUC de até 99%**

Os resultados foram comparáveis — e em alguns casos equivalentes — a modelos baseados em **CNNs** reportados na literatura, com **menor custo computacional**.

---

## 🆚 Comparação com CNNs

Diferente de abordagens baseadas em CNN:
- ❌ Não foi necessário data augmentation
- ❌ Não foi utilizado aprendizado por transferência
- ✅ Menor consumo de recursos computacionais
- ✅ Execução mais rápida

---

## 🏥 Aplicabilidade Clínica

O método proposto pode atuar como:
- Ferramenta auxiliar ao diagnóstico oftalmológico
- Apoio à detecção precoce da toxoplasmose ocular
- Solução de **baixo custo computacional**, ideal para ambientes com recursos limitados

---

## 📚 Conclusão

Os resultados demonstram que **características radiômicas combinadas com aprendizado de máquina** são eficazes para a classificação da toxoplasmose ocular, alcançando altos índices de desempenho mesmo com uma base de dados limitada.

O método proposto representa uma alternativa viável, eficiente e escalável para apoiar o diagnóstico médico e prevenir complicações graves como a cegueira.

---

## 🔮 Trabalhos Futuros

- Integração do método em sistemas clínicos reais
- Expansão da base de dados
- Implementação como serviço de apoio ao diagnóstico em redes de saúde
