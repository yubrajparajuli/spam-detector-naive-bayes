from pathlib import Path
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.model import SpamClassifier
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer

# paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "outputs" / "model"

def main():

    # step 1 - load data
    print("loading data...")
    loader = DataLoader(DATA_PATH)
    df = loader.load()
    df = loader.clean(df)
    loader.summary(df)

    # step 2 - preprocess
    print("\npreprocessing text...")
    preprocessor = TextPreprocessor()
    df = preprocessor.transform(df)
    print("preprocessing done:", df.shape)

    # step 3 - visualize eda
    print("\ngenerating eda plots...")
    viz = Visualizer(FIGURES_DIR)
    viz.plot_class_distribution(df)
    viz.plot_message_length(df)
    viz.plot_word_count(df)
    viz.plot_wordcloud(df)
    viz.plot_top_words(df)
    viz.plot_feature_comparison(df)
    viz.plot_correlation_heatmap(df)
    print("eda plots saved to:", FIGURES_DIR)

    # step 4 - train model
    print("\ntraining model...")
    X = df["text"]
    y = df["label"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classifier = SpamClassifier()
    classifier.tune(X_train, y_train)
    classifier.save(MODEL_DIR / "spam_detector.joblib")

    # step 5 - evaluate
    print("\nevaluating model...")
    evaluator = ModelEvaluator(classifier.model, X_test, y_test)
    evaluator.print_report()

    # step 6 - visualize results
    print("\ngenerating result plots...")
    cm = evaluator.confusion_matrix()
    fpr, tpr = evaluator.roc_curve()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, evaluator.y_prob)
    viz.plot_confusion_matrix(cm)
    viz.plot_roc_curve(fpr, tpr, auc)
    viz.plot_top_tfidf_features(classifier.model)
    print("result plots saved to:", FIGURES_DIR)

    # step 7 - cross validation
    print("\nrunning cross validation...")
    evaluator.cross_validate(X, y)

    # step 8 - error analysis
    print("\nerror analysis...")
    errors = evaluator.error_analysis(X_test)
    print("total errors:", len(errors))
    print(errors.head(10))

    print("\ndone! all outputs saved.")

if __name__ == "__main__":
    main()