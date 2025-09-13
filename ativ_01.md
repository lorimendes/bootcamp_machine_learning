### Bootcamp de Machine Learning
# Atividade 01

### 1. Explique, com suas palavras, o que é machine learning?
Aprendizado de máquina é um subconjunto de inteligência artificial que, a partir de uma grande volume de dados, é treinada para identificar padrões. Comm isso, ela gera probabilidades e previsões. Quanto mais dados de qualidade tiver, maior será a assertividade.

### 2. Explique o conceito de conjunto de treinamento, conjunto de validação e conjunto de teste em machine learning.
O conjunto de treinamento contém dados com rótulos que servirão para que o modelo identique padrões entre dados com mesmo rótulo, já o conjunto de validação fará a validação do modelo depois que passar pelo dados de treinamento. O conjunto de testes serve para uma avaliação final de um ciclo, após rodadas de treinamento e validação, para medir a acurácia do modelo. Os dados dos 3 conjuntos devem ser diferentes entre si para obter resultados de acurácia mais precisos.

### 3. Explique como você lidaria com dados ausentes em um conjunto de dados de treinamento.
Depende do volume do conjunto de dados, mas no caso de um base menor, pode ser feita apenas a remoção dos dados ausentes, caso sejam poucos e não deixe o conjunto desbalanceado quando comparado com as outras categorias. Se afetar o balanceamento no volume entre as classificações, é importante coletar mais dados para equilibrar. 

### 4. O que é uma matriz de confusão e como ela é usada para avaliar o desempenho de um modelo preditivo?
A matriz de confusão é uma tabela que representa e faz um cruzamento da distruição entre os positivos e negativos previstos pelo modelo e os reais de acordo com os rótulos para cada classficação. Essa matriz ajuda a denifir qual será o limite de classificação, a depender da quantidade de falsos positivos que queremos deixar passar, reduzindo o limite, ou a quantidade de falsos negativos que poderia perder, aumentando o limite de classificação.

### 5. Em quais áreas (tais como construção civil, agricultura, saúde, manufatura, entre outras) você acha mais interessante aplicar algoritmos de machine learning?
Dentre tantas possibilidades interessante, duas que me chamam a atenção, nesse momento, são: a aplicação em saúde para previsão dos riscos de doenças, e em sistemas de recomendação para personalização de experiências nos mais diversos serviços.
