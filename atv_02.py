## 1. Escreva uma função que receba uma lista de números e retorne outra lista com os números ímpares. 

def filtrar_impares (numeros):
  impares = []
  for numero in numeros:
    if numero % 2 != 0:
      impares.append(numero)
  return impares


## 2. Escreva uma função que receba uma lista de números e retorne outra lista com os números primos presentes. 

def verificar_primo(numero):
  if numero <= 1:
    return False
  for i in range(2, numero):
    if numero % i == 0:
      return False
    return True

def filtrar_primos(numeros):
  numeros_primos = []
  for numero in numeros:
    if verificar_primo(numero):
      numeros_primos.append(numero)
  return numeros_primos


## 3. Escreva uma função que receba duas listas e retorne outra lista com os elementos que estão presentes em apenas uma das listas. 

def filtrar_unicos(lista1, lista2):
  elementos_unicos = []
  for elemento in lista1:
    if elemento not in lista2:
      elementos_unicos.append(elemento)
  for elemento in lista2:
    if elemento not in lista1:
      elementos_unicos.append(elemento)
  return elementos_unicos


## 4. Dada uma lista de números inteiros, escreva uma função para encontrar o segundo maior valor na lista.

def encontrar_segundo_maior(lista):
  if len(lista) < 2:
    return "Lista com menos de 2 numeros."
  lista.sort(reverse=True)
  return lista[1]


## 5. Crie uma função que receba uma lista de tuplas, cada uma contendo o nome e a idade de uma pessoa, e retorne a lista ordenada pelo nome das pessoas em ordem alfabética.

def ordenar_por_nome(lista):
  lista_ordenada = sorted(lista, key=lambda pessoa: pessoa[0])
  return lista_ordenada

  
## 6. Como identificar e tratar outliers em uma coluna numérica usando desvio padrão ou quartis?

import pandas as pd
import numpy as np

def tratar_outliers_iqr(df, coluna):
  Q1 = df[coluna].quantile(0.25)
  Q3 = df[coluna].quantile(0.75)
  IQR = Q3 - Q1
   
  limite_superior = Q3 + 1.5 * IQR
  limite_inferior = Q1 - 1.5 * IQR

  df_tratado = df.copy()
  df_tratado[coluna] = np.where(df_tratado[coluna] > limite_superior, limite_superior, df_tratado[coluna])
    
  df_tratado[coluna] = np.where(df_tratado[coluna] < limite_inferior, limite_inferior, df_tratado[coluna])
    
  return df_tratado


## 7. Como concatenar vários DataFrames (empilhando linhas ou colunas), mesmo que tenham colunas diferentes? 
## Dica: Utiliza-se pd.concat() especificando axis=0 (linhas) ou axis=1 (colunas). Quando há colunas diferentes, os valores ausentes são preenchidos com NaN.

import pandas as pd

df_concatenado_linhas = pd.concat([df1, df2, df3], axis=0)
df_concatenado_colunas = pd.concat([df1, df2, df3], axis=1)


## 8. Utilizando pandas, como realizar a leitura de um arquivo CSV em um DataFrame e exibir as primeiras linhas?

import pandas as pd

df = pd.read_csv('arquivo.csv')
df_primeiras_linhas = df.head()


## 9.  Utilizando pandas, como selecionar uma coluna específica e filtrar linhas em um “DataFrame” com base em uma condição?

import pandas as pd

data = {'nome': ['Alice', 'Bob', 'Carlos', 'Daniel'],
        'idade': [25, 30, 22, 35],
        'cidade': ['Salvador', 'Rio', 'São Paulo', 'Recife']}
df = pd.DataFrame(data)

coluna_nome = df['nome']
df_filtrado_idade = df[df['idade'] > 25]
nomes_filtrados = df[df['idade'] > 25]['nome']


## 10.  Utilizando pandas, como lidar com valores ausentes (NaN) em um DataFrame?

import pandas as pd
import numpy as np

df_sem_nulos = df.dropna()
## ou
df_preenchido['coluna_A'].fillna(df_preenchido['coluna_A'].mean(), inplace=True)
df_preenchido['coluna_B'].fillna(df_preenchido['coluna_B'].mean(), inplace=True)
