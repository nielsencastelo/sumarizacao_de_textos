# -*- coding: utf-8 -*-
"""
# Sumarização de textos com bert

## Textos de exemplos
"""

from transformers import BartTokenizer, BartForConditionalGeneration

def resumir_texto(texto):
    # Inicializar o tokenizador e o modelo BART pré-treinado
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-mnli')

    # Preprocessar o texto
    inputs = tokenizer.encode(texto, return_tensors='pt')

    # Gerar o resumo
    summary_ids = model.generate(inputs, max_length=50, min_length=10, num_beams=4, early_stopping=True)
    resumo = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return resumo

# Exemplo de uso
texto_original = """A inteligência artificial é a inteligência similar à humana máquinas. 
                    Definem como o estudo de agente artificial com inteligência. 
                    Ciência e engenharia de produzir máquinas com inteligência. 
                    Resolver problemas e possuir inteligência. 
                    Relacionada ao comportamento inteligente. 
                    Construção de máquinas para raciocinar. 
                    Aprender com os erros e acertos. 
                    Inteligência artificial é raciocinar nas situações do cotidiano."""

resumo = resumir_texto(texto_original)
print(resumo)

