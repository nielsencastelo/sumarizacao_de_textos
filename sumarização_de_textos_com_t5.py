# -*- coding: utf-8 -*-
"""
# Sumarização de textos com T5

## Textos de exemplos
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text):
    # Inicializar o tokenizador e o modelo T5 pré-treinado
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Preprocessar o texto
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

    # Gerar a sumarização
    summary_ids = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Exemplo de uso
texto = """
A inteligência artificial é a inteligência similar à humana em máquinas.
Definem como o estudo de agente artificial com inteligência.
Ciência e engenharia de produzir máquinas com inteligência.
Resolver problemas e possuir inteligência.
Relacionada ao comportamento inteligente.
Construção de máquinas para raciocinar.
Aprender com os erros e acertos.
Inteligência artificial é raciocinar nas situações do cotidiano.
"""

resumo = summarize_text(texto)
print(resumo)


