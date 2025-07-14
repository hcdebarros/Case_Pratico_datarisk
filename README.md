# 📊 Caso DataRisk: Previsão de Inadimplência

Este repositório contém todo o pipeline de **Exploração**, **Engenharia de Features**, **Modelagem** e **Avaliação** de um projeto de **previsão de inadimplência** (risco de crédito), desenvolvido em Python com pandas e scikit-learn.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)  
- [Dados](#dados)  
- [Limpeza & EDA](#limpeza--eda)  
- [Engenharia de Features](#engenharia-de-features)  
- [Modelagem](#modelagem)  
- [Avaliação de Performance](#avaliação-de-performance)  
- [Requisitos](#requisitos)  


---

## 🔍 Visão Geral

Foi estimada a probabilidade de um cliente tornar-se **inadimplente** em um conjunto de títulos financeiros.  
- **Objetivo**: gerar uma coluna `PROBABILIDADE_INADIMPLENCIA` para cada registro de teste.  
- **Aplicação**: decision-making em concessão de crédito e priorização de cobrança.

---

## 🗄️ Dados

1. **base_cadastral.csv**  
   1.1 Base contendo informações cadastrais dos clientes. Cada cliente deve ter apenas uma data de cadastro e seus dados não mudam ao longo do tempo. 

* ID_CLIENTE: Identificador único do cliente.

* DATA_CADASTRO: Data da realização do cadastro no sistema. 

* DDD: Número do DDD do telefone do cliente.

* FLAG_PF: Indica se o cliente é uma pessoa física (‘X’) ou jurídica (‘NaN’). 

* SEGMENTO_INDUSTRIAL: Indica a qual segmento da indústria pertence o cliente. 

* DOMINIO_EMAIL: Indica o domínio(ou provedor) do email utilizado para o cadastro. 

* PORTE: Indica o porte (tamanho) da empresa.

* CEP_2_DIG: Indica os dois primeiros números do CEP do endereço cadastrado.

2. **base_info.csv**  
   - Base com informações adicionais dos clientes. Essa base é atualizada mensalmente, então cada cliente aparecerá apenas uma vez em cada mês de referência. Ou seja, o dentificador único da base consiste na combinação do ID_CLIENTE e da SAFRA_REF.

* ID_CLIENTE: Identificador único do cliente 

* SAFRA_REF: Mês de referência da amostra 

* RENDA_MES_ANTERIOR: Renda ou faturamento declarado pelo cliente no fim do mês anterior 
* NO_FUNCIONARIOS: Número de funcionários reportado pelo cliente no fim do mês anterior   

3. **base_pagamentos_desenvolvimento.csv**  
   - Base com informações históricas sobre transações (empréstimos) já realizadas pelos clientes. Cada cliente pode ter uma ou mais transações no mesmo período de tempo. Essa base deve ser utilizada no desenvolvimento do modelo, pois contém os dados de pagamento. 

* ID_CLIENTE: Identificador único do cliente

* SAFRA_REF: Mês de referência da amostra 

* DATA_EMISSAO_DOCUMENTO: Data da emissão da nota de crédito 

* DATA_VENCIMENTO: Data limite para pagamento do empréstimo 

* VALOR_A_PAGAR: Valor da nota de crédito 

* TAXA: Taxa de juros cobrada no empréstimo 

* DATA_PAGAMENTO: Data em que o cliente realizou o pagamento da nota de emissão, vencimento, pagamento e valor. 

4. **base_pagamentos_teste.csv**  
   - Base com as transações mais recentes, que ainda não foram pagas. Seu papel será prever a probabilidade de inadimplência para essas transações. A estrutura é a mesma da base de desenvolvimento, porém sem a coluna DATA_PAGAMENTO.

* ID_CLIENTE: Identificador único do cliente

* SAFRA_REF: Mês de referência da amostra 

* DATA_EMISSAO_DOCUMENTO: Data da emissão da nota de crédito 

* DATA_VENCIMENTO: Data limite para pagamento do empréstimo 

* VALOR_A_PAGAR: Valor da nota de crédito 

* TAXA: Taxa de juros cobrada no empréstimo 

Todas as bases foram unidas e processadas em `case_datarisk.ipynb`.

---

## 🧹 Limpeza & EDA

- **Tratamento de nulos** em `DDD` e `CEP_2_DIG` (imputação de categoria “Desconhecido” e valor constante).  
- **Conversão de datas** para o tipo datetime e posteriormente criação das features: `dias_desde_emissao` e `idade_cadastro_dias`.  
- **Exploração gráfica**: distribuições em log, boxplots por classe, correlações.  
- **Insights principais**:  
  - `VALOR_A_PAGAR` apresenta caudas longas — uso de escala log e winsorização.  
  - `NO_FUNCIONARIOS` é aproximadamente normal, mas com alguns outliers.  
  

---

## 🛠 Engenharia de Features

- **Winsorização (IQR capping)** em valores monetários e `NO_FUNCIONARIOS` para mitigar outliers.

- **Transformação log** (`np.log1p`) em `VALOR_A_PAGAR` para normalizar distribuição.

- **Imputação** de medianas em variáveis numéricas faltantes.  

- **Codificação**:  
  - Numéricas → `StandardScaler`  
  - Categóricas → `OneHotEncoder(handle_unknown='ignore')`

Todo o pipeline está implementado em `case_datarisk.ipynb`.

---

## 🤖 Modelagem

Testamos seis algoritmos clássicos:

| Modelo                 | AUC-ROC (validação) |
|------------------------|---------------------|
| Regressão Logística    | 0.7687              |
| Random Forest          | **0.9549**          |
| Gradient Boosting      | 0.8868              |
| AdaBoost               | 0.8205              |
| K-Nearest Neighbors    | 0.8796              |

- **Melhor modelo**: **RandomForestClassifier** (100 árvores, `random_state=42`).  
- Pipeline scikit-learn encadeando pré-processamento e classificador.

---

## 📈 Avaliação de Performance

Nesta etapa, medimos como cada modelo se saiu em separar clientes adimplentes de inadimplentes, utilizando o conjunto de **validação interna** (20 % da base de treino).

### Métricas principais

- **AUC-ROC** (Area Under ROC Curve)  
  Mede a capacidade geral de discriminar as duas classes em todos os thresholds possíveis.  
  - **Random Forest**: **0.9549**  
  - Gradient Boosting: 0.8868  
  - SVM (RBF): 0.8899  
  - K-Nearest Neighbors: 0.8796  
  - AdaBoost: 0.8205  
  - Logistic Regression: 0.7687  

- **Classification Report** (classe “1” – inadimplente):
  ```
                   precision    recall  f1-score   support
  inadimplente       0.9954    0.9963    0.9959      1087
  ```
  - **Precision**: 99.54 % (baixa taxa de falsos positivos)  
  - **Recall**:    99.63 % (capturou quase todos os inadimplentes)  
  - **F1-score**:  99.59 % (excelente equilíbrio entre precision e recall)

- **Matriz de Confusão**  
  ```
                         Previsto 0   Previsto 1
  Real 0 (adimplente)       14 391            5
  Real 1 (inadimplente)         4         1 083
  ```
  - **TP** = 1 083  
  - **FN** = 4  
  - **FP** = 5  
  - **TN** = 14 391  

### Interpretação

- **Erros mínimos**: apenas 9 amostras (0,06 % do total) classificadas incorretamente.  
- **Recall altíssimo** na classe minoritária, garantindo quase nenhum inadimplente “escapado”.  
- **Precision elevado**, com pouquíssimos alarmes falsos.  

> **Conclusão:** o **Random Forest** não só obteve a maior AUC, mas também entregou precision e recall excepcionais na classe crítica, tornando-o o modelo ideal para decisões de crédito e priorização de cobranças.

## 📦 Requisitos

- Estão disponíveis em 'Requiriments.txt'.
