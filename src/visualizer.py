# src/visualizer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path


class Visualizer:

    def __init__(self, figures_dir: str | Path):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["figure.dpi"] = 120

    def plot_class_distribution(self, df: pd.DataFrame) -> None:
        df["label"].value_counts().plot(kind="bar")
        plt.title("Class Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "class_distribution.png")
        plt.close()

    def plot_message_length(self, df: pd.DataFrame) -> None:
        df.boxplot(column="message_length", by="label")
        plt.title("Message Length by Label")
        plt.suptitle("")
        plt.xlabel("Label")
        plt.ylabel("Length")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "message_length_distribution.png")
        plt.close()

    def plot_word_count(self, df: pd.DataFrame) -> None:
        df.boxplot(column="word_count", by="label")
        plt.title("Word Count by Label")
        plt.suptitle("")
        plt.xlabel("Label")
        plt.ylabel("Word Count")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "word_count_distribution.png")
        plt.close()

    def plot_wordcloud(self, df: pd.DataFrame) -> None:
        for label, name in [(1, "spam"), (0, "ham")]:
            subset = df[df["label"] == label]["text"]
            words = " ".join(subset.dropna().tolist())
            if not words.strip():
                print(f"skipping wordcloud for {name} - no words found")
                continue
            wc = WordCloud(width=800, height=400, background_color="white").generate(words)
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{name.capitalize()} Words")
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"wordcloud_{name}.png")
            plt.close()

    def plot_top_words(self, df: pd.DataFrame, n: int = 20) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, label, name, color in zip(axes, [1, 0], ["spam", "ham"], ["red", "green"]):
            subset = df[df["label"] == label]["text"]
            words = " ".join(subset.dropna().tolist()).split()
            top_n = Counter(words).most_common(n)
            top_df = pd.DataFrame(top_n, columns=["word", "count"])
            top_df.plot(kind="bar", x="word", y="count", ax=ax, legend=False, color=color, alpha=0.7)
            ax.set_title(f"Top {n} {name.capitalize()} Words")
            ax.set_xlabel("Word")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "top_words.png")
        plt.close()

    def plot_feature_comparison(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col in zip(axes, ["punct_count", "digit_count", "word_count"]):
            df.boxplot(column=col, by="label", ax=ax)
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel("Label")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "feature_comparison.png")
        plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        corr = df[["label", "message_length", "word_count", "punct_count", "digit_count"]].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "correlation_heatmap.png")
        plt.close()

    def plot_confusion_matrix(self, cm) -> None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix.png")
        plt.close()

    def plot_roc_curve(self, fpr, tpr, auc) -> None:
        plt.plot(fpr, tpr, label=f"AUC = {round(auc, 3)}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "roc_curve.png")
        plt.close()

    def plot_top_tfidf_features(self, model, n: int = 20) -> None:
        feature_names = model.named_steps["tfidf"].get_feature_names_out()
        log_probs = model.named_steps["clf"].feature_log_prob_
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, label, idx, color in zip(
            axes, ["spam", "ham"], [1, 0], ["red", "green"]
        ):
            top_idx = log_probs[idx].argsort()[-n:][::-1]
            top_words = [(feature_names[i], log_probs[idx][i]) for i in top_idx]
            top_df = pd.DataFrame(top_words, columns=["word", "log_prob"])
            top_df.plot(kind="bar", x="word", y="log_prob", ax=ax, legend=False, color=color, alpha=0.7)
            ax.set_title(f"Top {n} {label.capitalize()} Features")
            ax.set_xlabel("Word")
            ax.set_ylabel("Log Probability")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "top_tfidf_features.png")
        plt.close()