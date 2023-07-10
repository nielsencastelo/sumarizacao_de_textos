# -*- coding: utf-8 -*-
"""
# Sumarização de textos com bert

## Textos de exemplos
"""

from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text):
    # Inicializar o tokenizador e o modelo BART pré-treinado
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-mnli')

    # Preprocessar o texto
    inputs = tokenizer.encode(text, return_tensors='pt')

    # Gerar a sumarização
    summary_ids = model.generate(inputs, max_length=150, min_length=40, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary

# Exemplo de uso
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisi. Mauris scelerisque tincidunt metus ac feugiat. Sed sit amet nisl id odio vestibulum accumsan. Donec iaculis finibus luctus. Pellentesque a velit cursus, bibendum lorem vel, fermentum sapien."
summary = summarize_text(text)
print(summary)
