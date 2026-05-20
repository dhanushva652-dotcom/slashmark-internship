# Task 1 — Sentiment Analysis

A complete internship-ready **sentiment analysis tool** that classifies text as **Positive / Neutral / Negative** for restaurant reviews, product feedback, or social posts.

## What this project includes
- Text preprocessing with **tokenization, stop-word removal, and lemmatization**
- **TF-IDF** feature extraction
- Classical ML models: **Logistic Regression, Linear SVM, Multinomial Naive Bayes**
- Evaluation using **accuracy, weighted F1-score, and classification report**
- A simple **Flask web app** for live predictions
- Support for **3 classes**: Negative (0), Neutral (1), Positive (2)

## How to run
```bash
pip install -r requirements.txt
python src/train.py
python run.py
```

Open: `http://127.0.0.1:5000`
