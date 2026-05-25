import joblib
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight


class SpamClassifier:

    def __init__(self, max_features: int = 5000, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=self.max_features)),
            ("clf", MultinomialNB())
        ])

    def tune(self, X_train, y_train) -> None:
        pipeline = self.build_pipeline()
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        param_grid = {"clf__alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train, clf__sample_weight=sample_weights)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print("best params:", self.best_params)
        print("best f1    :", round(grid_search.best_score_, 3))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)
        print("model saved:", path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        print("model loaded:", path)