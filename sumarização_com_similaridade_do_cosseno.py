# -*- coding: utf-8 -*-
"""


# Sumarização com similaridade do cosseno

## Preparação do texto de exemplo
"""

import re
import nltk
import string
import numpy as np
from nltk.cluster.util import cosine_distance
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('portuguese')
print(stopwords)

def preprocessamento(texto):
  texto_formatado = texto.lower()
  tokens = []
  for token in nltk.word_tokenize(texto_formatado):
    tokens.append(token)

  tokens = [palavra for palavra in tokens if palavra not in stopwords and palavra not in string.punctuation]
  texto_formatado = ' '.join([str(elemento) for elemento in tokens if not elemento.isdigit()])

  return texto_formatado

texto_original = """A inteligência artificial é a inteligência similar à humana máquinas. 
                    Definem como o estudo de agente artificial com inteligência. 
                    Ciência e engenharia de produzir máquinas com inteligência. 
                    Resolver problemas e possuir inteligência. 
                    Relacionada ao comportamento inteligente. 
                    Construção de máquinas para raciocinar. 
                    Aprender com os erros e acertos. 
                    Inteligência artificial é raciocinar nas situações do cotidiano."""
texto_original = re.sub(r'\s+', ' ', texto_original)
texto_original

"""## Função para calcular a similaridade entre sentenças

- Link: https://en.wikipedia.org/wiki/Cosine_similarity
- Cálculos passo a passo: https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
"""

sentencas_originais = [sentenca for sentenca in nltk.sent_tokenize(texto_original)]
sentencas_formatadas = [preprocessamento(sentenca_original) for sentenca_original in sentencas_originais]

sentencas_originais

sentencas_formatadas

def calcula_similaridade_sentencas(sentenca1, sentenca2):
  palavras1 = [palavra for palavra in nltk.word_tokenize(sentenca1)]
  palavras2 = [palavra for palavra in nltk.word_tokenize(sentenca2)]
  #print(palavras1)
  #print(palavras2)

  todas_palavras = list(set(palavras1 + palavras2))
  #print(todas_palavras)

  vetor1 = [0] * len(todas_palavras)
  vetor2 = [0] * len(todas_palavras)
  #print(vetor1)
  #print(vetor2)

  for palavra in palavras1:
    vetor1[todas_palavras.index(palavra)] += 1
  for palavra in palavras2:
    vetor2[todas_palavras.index(palavra)] += 1
  
  #print(vetor1)
  #print(vetor2) K-nn
  return 1 - cosine_distance(vetor1, vetor2)

calcula_similaridade_sentencas(sentencas_formatadas[0], sentencas_formatadas[0])

"""## Função para gerar a matriz de similaridade"""

def calcula_matriz_similaridade(sentencas):
  matriz_similaridade = np.zeros((len(sentencas), len(sentencas)))
  #print(matriz_similaridade)
  for i in range(len(sentencas)):
    for j in range(len(sentencas)):
      if i == j:
        continue
      matriz_similaridade[i][j] = calcula_similaridade_sentencas(sentencas[i], sentencas[j])

  return matriz_similaridade

calcula_matriz_similaridade(sentencas_formatadas)

"""## Função para sumarizar

- Algoritmo Page Rank: https://www.youtube.com/watch?v=YfDNI1jp5sM e https://www.youtube.com/watch?v=YplmCue8XJU
"""

def sumarizar(texto, quantidade_sentencas):
  sentencas_originais = [sentenca for sentenca in nltk.sent_tokenize(texto)]
  sentencas_formatadas = [preprocessamento(sentenca_original) for sentenca_original in sentencas_originais]
  matriz_similaridade = calcula_matriz_similaridade(sentencas_formatadas)
  #print(matriz_similaridade)
  grafo_similaridade = nx.from_numpy_array(matriz_similaridade)
  #print(grafo_similaridade.nodes)
  #print(grafo_similaridade.edges)
  notas = nx.pagerank(grafo_similaridade)
  #print(notas)
  notas_ordenadas = sorted(((notas[i], nota) for i, nota in enumerate(sentencas_originais)), reverse=True)
  #print(notas_ordenadas)
  melhores_sentencas = []
  for i in range(quantidade_sentencas):
    melhores_sentencas.append(notas_ordenadas[i][1])
  
  return sentencas_originais, melhores_sentencas, notas_ordenadas

sentencas_originais, melhores_sentencas, notas_sentencas = sumarizar(texto_original, 3)

sentencas_originais

melhores_sentencas

notas_sentencas

"""## Visualização do resumo"""

def visualiza_resumo(titulo, lista_sentencas, melhores_sentencas):
  from IPython.core.display import HTML
  texto = ''

  display(HTML(f'<h1>Resumo do texto - {titulo}</h1>'))
  for i in lista_sentencas:
    if i in melhores_sentencas:
      texto += str(i).replace(i, f"<mark>{i}</mark>")
    else:
      texto += i
  display(HTML(f""" {texto} """))

visualiza_resumo('Teste', sentencas_originais, melhores_sentencas)

"""## Extração de texto da internet"""

#!pip install goose3

from goose3 import Goose
g = Goose()
url = 'https://iaexpert.academy/2020/11/09/ia-preve-resultado-das-eleicoes-americanas/'
artigo = g.extract(url)

artigo.cleaned_text

sentencas_originais, melhores_sentencas, notas_sentencas = sumarizar(artigo.cleaned_text, 5)

sentencas_originais

melhores_sentencas

notas_sentencas

visualiza_resumo('Teste', sentencas_originais, melhores_sentencas)

"""## Solução para o exercício - lematização"""

import spacy

#!python -m spacy download pt

pln = spacy.load('pt')
pln

def preprocessamento_lematizacao(texto):
  texto = texto.lower()
  texto = re.sub(r" +", ' ', texto)

  documento = pln(texto)
  tokens = []
  for token in documento:
    tokens.append(token.lemma_)
  
  tokens = [palavra for palavra in tokens if palavra not in stopwords and palavra not in string.punctuation]
  texto_formatado = ' '.join([str(elemento) for elemento in tokens if not elemento.isdigit()])
  
  return texto_formatado

def sumarizar_lematizacao(texto, quantidade_sentencas):
  sentencas_originais = [sentenca for sentenca in nltk.sent_tokenize(texto)]
  sentencas_formatadas = [preprocessamento_lematizacao(sentenca_original) for sentenca_original in sentencas_originais]
  matriz_similaridade = calcula_matriz_similaridade(sentencas_formatadas)
  grafo_similaridade = nx.from_numpy_array(matriz_similaridade)
  notas = nx.pagerank(grafo_similaridade)
  notas_ordenadas = sorted(((notas[i], nota) for i, nota in enumerate(sentencas_originais)), reverse=True)    
  melhores_sentencas = []
  for i in range(quantidade_sentencas):
    melhores_sentencas.append(notas_ordenadas[i][1])

  return sentencas_originais, melhores_sentencas, notas_ordenadas

artigo.cleaned_text

sentencas_originais, melhores_sentencas, _ = sumarizar(artigo.cleaned_text, 5)
visualiza_resumo(artigo.title, sentencas_originais, melhores_sentencas)

sentencas_originais, melhores_sentencas, _ = sumarizar_lematizacao(artigo.cleaned_text, 5)
visualiza_resumo(artigo.title, sentencas_originais, melhores_sentencas)