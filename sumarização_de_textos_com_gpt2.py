# -*- coding: utf-8 -*-
"""
# Sumarização de textos com bert

## Textos de exemplos
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def summarize_text(text):
    # Inicializar o tokenizador e o modelo GPT-2 pré-treinado
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenizar o texto
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Gerar a sumarização
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

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

