import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class FeatureEngineer:

    def __init__(self, max_features: int = 5000, test_size: float = 0.2, random_state: int = 42):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(max_features=self.max_features)

    def split(self, df: pd.DataFrame):
        X = df["text"]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train, X_test):
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def get_feature_names(self):
        return self.tfidf.get_feature_names_out()