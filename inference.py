import torch
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input["input_ids"])
output = model(**encoded_input)

print(output["last_hidden_state"].shape)