# Spam Detector — Naive Bayes

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-yubraj101%2Fspam--detector-blue)
![License](https://img.shields.io/badge/License-MIT-green)

SMS spam detection using Naive Bayes and TF-IDF vectorization trained on the UCI SMS Spam Collection dataset.

## How It Works

1. raw SMS messages are cleaned and lemmatized
2. TF-IDF converts text into numerical features
3. MultinomialNB trained with class balancing and hyperparameter tuning
4. outputs metrics, plots, and saved model

## Tech Stack

`scikit-learn` `pandas` `numpy` `nltk` `matplotlib` `seaborn` `wordcloud` `joblib`

## Dataset

SMS Spam Collection — UCI ML Repository (5169 messages, 13.5% spam)

## Results

| Metric     | Score |
| ---------- | ----- |
| Accuracy   | 96.3% |
| Precision  | 81.6% |
| Recall     | 91.6% |
| F1 Score   | 86.3% |
| ROC AUC    | 99.2% |
| CV F1 Mean | 91.2% |

## Key Findings

- digit count has 0.84 correlation with spam — strongest single feature
- spam messages average 138 chars vs 71 for ham — consistent pattern
- top spam words: `call`, `free`, `claim`, `urgent`, `prize`, `win`

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

### local

```bash
python main.py
```

### notebook

```bash
jupyter notebook notebook/spam_detection_naive_bayes.ipynb
```

### docker

```bash
docker pull yubraj101/spam-detector
docker run --rm -v "$(pwd)/outputs:/app/outputs" yubraj101/spam-detector
```

## License

MIT
