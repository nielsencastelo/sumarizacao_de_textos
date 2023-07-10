# -*- coding: utf-8 -*-
"""Algoritmo básico baseado em frequência para sumarização

# Algoritmo básico baseado em frequência

## Pré-processamento do texto
"""

import re
import nltk
import string

texto_original = """A inteligência artificial é a inteligência similar à humana. 
                    Definem como o estudo de agente artificial com inteligência. 
                    Ciência e engenharia de produzir máquinas com inteligência. 
                    Resolver problemas e possuir inteligência. 
                    Relacionada ao comportamento inteligente. 
                    Construção de máquinas para raciocinar. 
                    Aprender com os erros e acertos. 
                    Inteligência artificial é raciocinar nas situações do cotidiano."""

texto_original

texto_original = re.sub(r'\s+', ' ', texto_original)

texto_original

nltk.download('punkt')

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('portuguese')
print(stopwords)

len(stopwords)

string.punctuation

def preprocessamento(texto):
  texto_formatado = texto.lower()
  tokens = []
  for token in nltk.word_tokenize(texto_formatado):
    tokens.append(token)

  tokens = [palavra for palavra in tokens if palavra not in stopwords and palavra not in string.punctuation]
  texto_formatado = ' '.join([str(elemento) for elemento in tokens if not elemento.isdigit()])

  return texto_formatado

texto_formatado = preprocessamento(texto_original)
texto_formatado

"""## Frequência das palavras"""

frequencia_palavras = nltk.FreqDist(nltk.word_tokenize(texto_formatado))
frequencia_palavras

frequencia_palavras['situações']

frequencia_palavras.keys()

frequencia_maxima = max(frequencia_palavras.values())

frequencia_maxima

for palavra in frequencia_palavras.keys():
  frequencia_palavras[palavra] = (frequencia_palavras[palavra] / frequencia_maxima)

frequencia_palavras

"""## Tokenização de sentenças"""

'o dr. joão foi para casa. Ele chegou cedo'.split('.')

nltk.word_tokenize('o dr. joão foi para casa. Ele chegou cedo')

nltk.sent_tokenize('o dr. joão foi para casa. Ele chegou cedo')

lista_sentencas = nltk.sent_tokenize(texto_original)
lista_sentencas

"""## Geração do resumo (nota para as sentenças)"""

frequencia_palavras

nota_sentencas = {}
for sentenca in lista_sentencas:
  #print(sentenca)
  for palavra in nltk.word_tokenize(sentenca.lower()):
    #print(palavra)
    if palavra in frequencia_palavras.keys():
      if sentenca not in nota_sentencas.keys():
        nota_sentencas[sentenca] = frequencia_palavras[palavra]
      else:
        nota_sentencas[sentenca] += frequencia_palavras[palavra]

nota_sentencas

import heapq
melhores_sentencas = heapq.nlargest(3, nota_sentencas, key=nota_sentencas.get)

melhores_sentencas

resumo = ' '.join(melhores_sentencas)
resumo

texto_original

"""## Visualização do resumo"""

from IPython.core.display import HTML
texto = ''

display(HTML(f'<h1>Resumo do texto</h1>'))
for sentenca in lista_sentencas:
  #texto += sentenca
  if sentenca in melhores_sentencas:
    texto += str(sentenca).replace(sentenca, f"<mark>{sentenca}</mark>")
  else:
    texto += sentenca
display(HTML(f"""{texto}"""))

"""## Extração de texto da internet"""

#!pip install goose3

from goose3 import Goose

g = Goose()
url = 'https://iaexpert.academy/2020/11/09/ia-preve-resultado-das-eleicoes-americanas/'
artigo = g.extract(url)

artigo.infos

artigo.title

artigo.links

artigo.tags

artigo.cleaned_text

len(artigo.cleaned_text)

artigo_original = artigo.cleaned_text
artigo_original

artigo_formatado = preprocessamento(artigo_original)
artigo_formatado

len(artigo_formatado)

def sumarizar(texto, quantidade_sentencas):
  texto_original = texto
  texto_formatado = preprocessamento(texto_original)

  frequencia_palavras = nltk.FreqDist(nltk.word_tokenize(texto_formatado))
  frequencia_maxima = max(frequencia_palavras.values())
  for palavra in frequencia_palavras.keys():
    frequencia_palavras[palavra] = (frequencia_palavras[palavra] / frequencia_maxima)
  lista_sentencas = nltk.sent_tokenize(texto_original)
  
  nota_sentencas = {}
  for sentenca in lista_sentencas:
    for palavra in nltk.word_tokenize(sentenca):
      if palavra in frequencia_palavras.keys():
        if sentenca not in nota_sentencas.keys():
          nota_sentencas[sentenca] = frequencia_palavras[palavra]
        else:
          nota_sentencas[sentenca] += frequencia_palavras[palavra]

  import heapq
  melhores_sentencas = heapq.nlargest(quantidade_sentencas, nota_sentencas, key=nota_sentencas.get)

  return lista_sentencas, melhores_sentencas, frequencia_palavras, nota_sentencas

lista_sentencas, melhores_sentencas, frequencia_palavras, nota_sentencas = sumarizar(artigo_original, 5)

lista_sentencas, len(lista_sentencas)

melhores_sentencas

frequencia_palavras

nota_sentencas

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

visualiza_resumo('Eleições', lista_sentencas, melhores_sentencas)

"""## Sumarização de mais textos"""

lista_artigos = ['https://iaexpert.academy/2020/11/06/ia-detecta-deep-fakes-produzidos-com-tecnicas-recentes/',
                 'https://iaexpert.academy/2020/11/13/facebook-apresenta-novo-algoritmo-deteccao-fake-news/',
                 'https://iaexpert.academy/2020/11/16/automl-aspectos-aplicacoes/']

for url in lista_artigos:
  #print(url)
  g = Goose()
  artigo = g.extract(url)
  lista_sentencas, melhores_sentencas, _, _ = sumarizar(artigo.cleaned_text, 5)
  visualiza_resumo(artigo.title, lista_sentencas, melhores_sentencas)

"""## Solução para o exercício - lematização"""

import spacy

#!python -m spacy download pt

pln = spacy.load('pt')
pln

documento = pln('inteligentes inteligente inteligência corrida corrido correr correndo correstes')
for token in documento:
  print(token.text, token.lemma_)

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

preprocessamento_lematizacao(texto_original)

texto_original

def sumarizar_lematizacao(texto, quantidade_sentencas):
  texto_original = texto
  # Chamada para a outra função de pré-processamento
  texto_formatado = preprocessamento_lematizacao(texto_original)

  frequencia_palavras = nltk.FreqDist(nltk.word_tokenize(texto_formatado))
  frequencia_maxima = max(frequencia_palavras.values())
  for palavra in frequencia_palavras.keys():
    frequencia_palavras[palavra] = (frequencia_palavras[palavra] / frequencia_maxima)
  lista_sentencas = nltk.sent_tokenize(texto_original)
  
  nota_sentencas = {}
  for sentenca in lista_sentencas:
    for palavra in nltk.word_tokenize(sentenca):
      if palavra in frequencia_palavras.keys():
        if sentenca not in nota_sentencas.keys():
          nota_sentencas[sentenca] = frequencia_palavras[palavra]
        else:
          nota_sentencas[sentenca] += frequencia_palavras[palavra]

  import heapq
  melhores_sentencas = heapq.nlargest(quantidade_sentencas, nota_sentencas, key=nota_sentencas.get)

  return lista_sentencas, melhores_sentencas, frequencia_palavras, nota_sentencas

for url in lista_artigos:
  #print(url)
  g = Goose()
  artigo = g.extract(url)
  lista_sentencas, melhores_sentencas, _, _ = sumarizar_lematizacao(artigo.cleaned_text, 5)
  visualiza_resumo(artigo.title, lista_sentencas, melhores_sentencas)