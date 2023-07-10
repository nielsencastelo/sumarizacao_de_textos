# -*- coding: utf-8 -*-
"""
# Sumarização de textos com bibliotecas

## Textos de exemplos
"""

import nltk
nltk.download('punkt')

#!pip install goose3

from goose3 import Goose
g = Goose()
url = 'https://iaexpert.academy/2020/11/09/ia-preve-resultado-das-eleicoes-americanas/'
artigo_portugues = g.extract(url)

url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'
artigo_ingles = g.extract(url)

artigo_portugues.cleaned_text

artigo_ingles.cleaned_text

"""## Biblioteca sumy

- https://pypi.org/project/sumy/
"""

#!pip install sumy

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text, Tokenizer('portuguese'))

sumarizador = LuhnSummarizer()

resumo = sumarizador(parser.document, 5)

resumo

for sentenca in resumo:
  print(sentenca)

"""## Biblioteca pysummarization

- https://pypi.org/project/pysummarization/
- Redes neurais recorrentes: https://www.youtube.com/watch?v=ZvBJxh5O3H0
"""

#!pip install pysummarization

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

auto_abstractor = AutoAbstractor()
auto_abstractor.tokenizable_doc = SimpleTokenizer()
auto_abstractor.delimiter_list = [".", "\n"]
abstractable_doc = TopNRankAbstractor()
resumo = auto_abstractor.summarize(artigo_ingles.cleaned_text, abstractable_doc)

resumo

for sentenca in resumo['summarize_result']:
  print(sentenca)

"""## Biblioteca - BERT

- https://pypi.org/project/bert-extractive-summarizer/
- Arquitetura BERT: https://www.youtube.com/watch?v=ERA1bjBKqtE
"""

#!pip install bert-extractive-summarizer

from summarizer import Summarizer

sumarizador = Summarizer()
resumo = sumarizador(artigo_ingles.cleaned_text)

resumo

len(artigo_ingles.cleaned_text), len(resumo)

"""## Solução para o exercício - biblioteca sumy"""

sentencas_originais = [sentenca for sentenca in nltk.sent_tokenize(artigo_portugues.cleaned_text)]
sentencas_originais

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.reduction import ReductionSummarizer

"""### LsaSummarizer"""

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text,Tokenizer('portuguese'))
sumarizador = LsaSummarizer()
resumo = sumarizador(parser.document, 5)
melhores_sentencas = []
for sentenca in resumo:
  print(sentenca)

"""### LexRankSummarizer"""

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text,Tokenizer('portuguese'))
sumarizador = LexRankSummarizer()
resumo = sumarizador(parser.document, 5)
melhores_sentencas = []
for sentenca in resumo:
  print(sentenca)

"""### TextRankSummarizer"""

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text,Tokenizer('portuguese'))
sumarizador = TextRankSummarizer()
resumo = sumarizador(parser.document, 5)
melhores_sentencas = []
for sentenca in resumo:
  print(sentenca)

"""### SumBasicSummarizer"""

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text,Tokenizer('portuguese'))
sumarizador = SumBasicSummarizer()
resumo = sumarizador(parser.document, 5)
melhores_sentencas = []
for sentenca in resumo:
  print(sentenca)

"""### KLSummarizer"""

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text,Tokenizer('portuguese'))
sumarizador = KLSummarizer()
resumo = sumarizador(parser.document, 5)
melhores_sentencas = []
for sentenca in resumo:
  print(sentenca)

"""### ReductionSummarizer"""

parser = PlaintextParser.from_string(artigo_portugues.cleaned_text,Tokenizer('portuguese'))
sumarizador = ReductionSummarizer()
resumo = sumarizador(parser.document, 5)
melhores_sentencas = []
for sentenca in resumo:
  print(sentenca)