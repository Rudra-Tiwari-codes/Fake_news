# ==================== INSTALL DEPENDENCIES ====================
# !pip install transformers datasets wordcloud gdown

# ==================== IMPORTS ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import torch
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

# ==================== DOWNLOADS ====================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==================== DATA INGESTION ====================
# Upload and unzip dataset
# !gdown "https://drive.google.com/uc?id=178f_VkNxccNidap-5-uffXUW475pAuPy&confirm=t"
# !unzip fake-news.zip

news_d = pd.read_csv("train.csv")
print("Shape of News data:", news_d.shape)
print(news_d.head())

# ==================== TEXT ANALYTICS ====================
txt_length = news_d.text.str.split().str.len()
title_length = news_d.title.str.split().str.len()
print(txt_length.describe())
print(title_length.describe())

sns.countplot(x="label", data=news_d)
plt.title("Label Distribution")
plt.show()
print(news_d.label.value_counts(normalize=True) * 100)

# ==================== DATA CLEANING ====================
remove_c = ['id','author']
text_f = ['title', 'text']
ps = WordNetLemmatizer()
stopwords_dict = Counter(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http[\w:/\.]+", " ", str(text))
    text = re.sub(r"[^\.\w\s]", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s\s+", " ", text)
    return text.lower().strip()

def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    return ' '.join([ps.lemmatize(word) for word in wordlist if word not in stopwords_dict])

def clean_dataset(df):
    df = df.drop(remove_c, axis=1)
    df[text_f] = df[text_f].fillna("None")
    df["text"] = df["text"].apply(nltk_preprocess)
    df["title"] = df["title"].apply(nltk_preprocess)
    return df

df = clean_dataset(news_d)

# ==================== WORDCLOUDS ====================
def generate_wordcloud(text, title):
    wordcloud = WordCloud(background_color='black', width=800, height=600).generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_wordcloud(' '.join(df[df.label==0]['text']), "Reliable News WordCloud")
generate_wordcloud(' '.join(df[df.label==1]['text']), "Fake News WordCloud")

# ==================== TOP N-GRAMS ====================
def plot_top_ngrams(corpus, title, ylabel, n=2):
    ngram_freq = pd.Series(nltk.ngrams(corpus.split(), n)).value_counts()[:20]
    ngram_freq.sort_values().plot.barh(figsize=(12, 8))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Frequency")
    plt.show()

plot_top_ngrams(' '.join(df[df.label==0]['text']), "Top Bigrams - Reliable", "Bigrams", n=2)
plot_top_ngrams(' '.join(df[df.label==1]['text']), "Top Bigrams - Fake", "Bigrams", n=2)

# ==================== BERT TRAINING PREP ====================
model_name = "bert-base-uncased"
max_length = 512

tokenizer = BertTokenizerFast.from_pretrained(model_name)

news_df = news_d.dropna(subset=["text", "title", "author"])

def prepare_data(df, test_size=0.2):
    texts, labels = [], []
    for _, row in df.iterrows():
        text = f"{row['author']} : {row['title']} - {row['text']}"
        texts.append(text)
        labels.append(row["label"])
    return train_test_split(texts, labels, test_size=test_size)

train_texts, valid_texts, train_labels, valid_labels = prepare_data(news_df)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
valid_dataset = NewsDataset(valid_encodings, valid_labels)

# ==================== MODEL & TRAINING ====================
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    warmup_steps=100,
    logging_dir='./logs',
    logging_steps=200,
    save_steps=200,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# ==================== SAVE MODEL ====================
model_path = "fake-news-bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# ==================== INFERENCE ====================
def get_prediction(text, convert_to_label=False):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return ["reliable", "fake"][pred] if convert_to_label else pred

# ==================== TEST SET SUBMISSION ====================
test_df = pd.read_csv("test.csv")
test_df["new_text"] = test_df["author"].astype(str) + " : " + test_df["title"].astype(str) + " - " + test_df["text"].astype(str)
test_df["label"] = test_df["new_text"].apply(get_prediction)
test_df[["id", "label"]].to_csv("submit_final.csv", index=False)
