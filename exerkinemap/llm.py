from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Fine-tuning Example
inputs = tokenizer("This is a sample input.", return_tensors='pt')
labels = torch.tensor([1]).unsqueeze(0)
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# Model selection based on performance and tasks
