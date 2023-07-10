# -*- coding: utf-8 -*-
"""

# Sumarização de textos com o Algoritmo de Luhn

- https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf

## Preparação do texto de exemplo
"""

import re
import nltk
import string
import heapq

nltk.download('punkt')

nltk.download('stopwords')

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

stopwords = nltk.corpus.stopwords.words('portuguese')
print(stopwords)

stopwords.append('ser')
stopwords.append('além')

print(stopwords)

def preprocessamento(texto):
  texto_formatado = texto.lower()
  tokens = []
  for token in nltk.word_tokenize(texto_formatado):
    tokens.append(token)

  tokens = [palavra for palavra in tokens if palavra not in stopwords and palavra not in string.punctuation]
  texto_formatado = ' '.join([str(elemento) for elemento in tokens if not elemento.isdigit()])

  return texto_formatado

"""## Função para calcular a nota das sentenças"""

teste = ['a', 'b', 'c']
teste.index('g')

def calcula_nota_sentenca(sentencas, palavras_importantes, distancia):
  notas = []
  indice_sentenca = 0

  for sentenca in [nltk.word_tokenize(sentenca.lower()) for sentenca in sentencas]:
    #print('---------------')
    #print(sentenca)
    indice_palavra = []
    for palavra in palavras_importantes:
      #print(palavra)
      try:
        indice_palavra.append(sentenca.index(palavra))
      except ValueError:
        pass
    
    indice_palavra.sort()
    #print(indice_palavra)

    if len(indice_palavra) == 0:
      continue

    # [0, 1, 3, 5]
    lista_grupos = []
    grupo = [indice_palavra[0]]
    i = 1
    while i < len(indice_palavra):
      if indice_palavra[i] - indice_palavra[i - 1] < distancia:
        grupo.append(indice_palavra[i])
        #print('grupo: ', grupo)
      else:
        lista_grupos.append(grupo[:])
        grupo = [indice_palavra[i]]
        #print('grupo: ', grupo)
      i += 1
    lista_grupos.append(grupo)
    #print('todos os grupos: ', lista_grupos)

    nota_maxima_grupo = 0
    for g in lista_grupos:
      #print(g)
      palavras_importantes_no_grupo = len(g)
      total_palavras_no_grupo = g[-1] - g[0] + 1
      #print('palavras importantes no grupo ', palavras_importantes_no_grupo)
      #print('total de palavras ', total_palavras_no_grupo)
      nota = 1.0 * palavras_importantes_no_grupo**2 / total_palavras_no_grupo
      #print('nota grupo', nota)

      if nota > nota_maxima_grupo:
        nota_maxima_grupo = nota

    notas.append((nota_maxima_grupo, indice_sentenca))
    indice_sentenca += 1

  #print('notas finais das senteças', notas)
  return notas

teste = [0, 1, 3, 4, 6, 9]
teste[-1], teste[0] + 1

"""## Função para sumarizar os textos"""

def sumarizar(texto, top_n_palavras, distancia, quantidade_sentencas):
  sentencas_originais = [sentenca for sentenca in nltk.sent_tokenize(texto)]
  #print(sentencas_originais)
  sentencas_formatadas = [preprocessamento(sentenca_original) for sentenca_original in sentencas_originais]
  #print(sentencas_formatadas)
  palavras = [palavra for sentenca in sentencas_formatadas for palavra in nltk.word_tokenize(sentenca)]
  #print(palavras)
  frequencia = nltk.FreqDist(palavras)
  #return frequencia
  top_n_palavras = [palavra[0] for palavra in frequencia.most_common(top_n_palavras)]
  #print(top_n_palavras)
  notas_sentencas = calcula_nota_sentenca(sentencas_formatadas, top_n_palavras, distancia)
  #print(notas_sentencas)
  melhores_sentencas = heapq.nlargest(quantidade_sentencas, notas_sentencas)
  #print(melhores_sentencas)
  melhores_sentencas = [sentencas_originais[i] for (nota, i) in melhores_sentencas]
  #print(melhores_sentencas)
  #print(sentencas_originais)
  return sentencas_originais, melhores_sentencas, notas_sentencas

sentencas_originais, melhores_sentencas, notas_sentencas = sumarizar(texto_original, 5, 3, 3)

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

sentencas_originais, melhores_sentencas, notas_sentencas = sumarizar(artigo.cleaned_text, 300, 10, 5)

sentencas_originais

melhores_sentencas

notas_sentencas

visualiza_resumo(artigo.title, sentencas_originais, melhores_sentencas)

"""## Leitura de artigos de feed de notícias (RSS)"""

#!pip install feedparser

import feedparser

from bs4 import BeautifulSoup
import os
import json

url = 'https://iaexpert.academy/feed/'
feed = feedparser.parse(url)

feed.entries

for e in feed.entries:
  print(e.title)
  print(e.links[0].href)
  print(e.content[0].value)

e.content[0].value

def limpa_html(texto):
  if texto == '':
    return ''
  return BeautifulSoup(texto, 'html5lib').get_text()

limpa_html(e.content[0].value)

artigos = []
for e in feed.entries:
  artigos.append({'titulo': e.title, 'conteudo': limpa_html(e.content[0].value)})

artigos

arquivo_gravar = os.path.join('feed_iaexpert.json')
arquivo = open(arquivo_gravar, 'w+')
arquivo.write(json.dumps(artigos, indent=1))
arquivo.close()

artigos_blog = json.loads(open('/content/feed_iaexpert.json').read())
artigos_blog

"""## Nuvem de palavras"""

artigos_blog[0]['titulo']

conteudo_feed = ''
for artigo in artigos_blog:
  conteudo_feed += artigo['conteudo']

conteudo_feed

conteudo_feed_formatado = preprocessamento(conteudo_feed)
conteudo_feed_formatado

len(conteudo_feed), len(conteudo_feed_formatado)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(WordCloud().generate(conteudo_feed_formatado))

"""## Extração de entidades nomeadas

- Siglas: https://spacy.io/api/annotation#named-entities
"""

import spacy

#!python -m spacy download pt

pln = spacy.load('pt')
pln

documento = pln(conteudo_feed_formatado)

from spacy import displacy
displacy.render(documento, style = 'ent', jupyter = True)

for entidade in documento.ents:
  if entidade.label_ == 'LOC':
    print(entidade.text, entidade.label_)

"""## Sumarização de artigos de feed de notícias"""

for artigo in artigos_blog:
  #print(artigo['titulo'])
  #print(artigo['conteudo'])
  sentencas_originais, melhores_sentencas, _ = sumarizar(artigo['conteudo'], 150, 10, 5)
  visualiza_resumo(artigo['titulo'], sentencas_originais, melhores_sentencas)
  salva_resumo(artigo['titulo'], sentencas_originais, melhores_sentencas)

"""## Geração de arquivos HTML"""

def salva_resumo(titulo, lista_sentencas, melhores_sentencas):
  HTML_TEMPLATE = """<html>
    <head>
      <title>{0}</title>
      <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    </head>
    <body>{1}</body>

  </html>"""
  texto = ''
  for i in lista_sentencas:
    if i in melhores_sentencas:
      texto += str(i).replace(i, f"<mark>{i}</mark>")
    else:
      texto += i

  arquivo = open(os.path.join(titulo + '.html'), 'wb')
  html = HTML_TEMPLATE.format(titulo + ' - resumo', texto)
  arquivo.write(html.encode('utf-8'))
  arquivo.close()

"""## Solução para o exercício - lematização"""

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

def sumarizar_lematizacao(texto, top_n_palavras, distancia, quantidade_sentencas):
  sentencas_originais = [sentenca for sentenca in nltk.sent_tokenize(texto)]
  sentencas_formatadas = [preprocessamento_lematizacao(sentenca_original) for sentenca_original in sentencas_originais]
  palavras = [palavra.lower() for sentenca in sentencas_formatadas for palavra in nltk.tokenize.word_tokenize(sentenca)]
  frequencia = nltk.FreqDist(palavras)
  top_n_palavras = [palavra[0] for palavra in frequencia.most_common(top_n_palavras)]
  notas_sentencas = calcula_nota_sentenca(sentencas_formatadas, top_n_palavras, distancia)
  melhores_sentencas = heapq.nlargest(quantidade_sentencas, notas_sentencas)
  melhores_sentencas = [sentencas_originais[i] for (nota, i) in melhores_sentencas]
  
  return sentencas_originais, melhores_sentencas, notas_sentencas

artigos_blog[0]['conteudo']

sentencas_originais, melhores_sentencas, _ = sumarizar(artigos_blog[0]['conteudo'], 300, 5, 5)
visualiza_resumo(artigos_blog[0]['titulo'], sentencas_originais, melhores_sentencas)

sentencas_originais, melhores_sentencas, _ = sumarizar_lematizacao(artigos_blog[0]['conteudo'], 300, 5, 5)
visualiza_resumo(artigos_blog[0]['titulo'], sentencas_originais, melhores_sentencas)