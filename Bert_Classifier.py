# Install required library
# !pip install transformers datasets

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

# Upload your CSV file in Google Colab
from google.colab import files
uploaded = files.upload()

# Load dataset
path = "fake_news_data.csv"
data = pd.read_csv(path)

# Split dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Tokenize inputs
train_encodings = tokenizer(list(train_data["text"]), truncation=True, padding=True, return_tensors="pt")
test_encodings = tokenizer(list(test_data["text"]), truncation=True, padding=True, return_tensors="pt")

# Convert labels
train_labels = torch.tensor(train_data["label"].values)
test_labels = torch.tensor(test_data["label"].values)

# Create dataloaders
train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop (1 epoch for demo)
model.train()
for batch in train_loader:
    input_ids, attention_mask, labels = [x.to(device) for x in batch]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Evaluation
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, axis=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")
