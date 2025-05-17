Fake News Detection with BERT

This script trains a BERT-based model to classify whether news articles are real or fake using the Hugging Face Transformers library.
- Uses pre-trained `bert-base-cased` model
- Tokenizes and prepares text data with Hugging Face Tokenizer
- Fine-tunes the model on a custom labeled dataset
- Outputs model accuracy on a test set
## 📰 Fake News Detection with BERT

A comprehensive NLP pipeline that performs fake news detection using BERT and Hugging Face Transformers. This project includes data cleaning, visualization, training, and inference.

### 📁 File: `bert_fake_news_pipeline.py`

### ✅ Features
- Cleans and preprocesses news articles (removes stopwords, applies lemmatization)
- Generates visualizations (word clouds, n-gram histograms)
- Fine-tunes a `bert-base-uncased` model using Hugging Face's `Trainer`
- Predicts on test data and prepares submission file
