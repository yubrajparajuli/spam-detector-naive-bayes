import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd


class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def to_lowercase(self, text: str) -> str:
        return text.lower()

    def strip_whitespace(self, text: str) -> str:
        return text.strip()

    def remove_extra_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def remove_tabs_and_newlines(self, text: str) -> str:
        return text.replace("\t", " ").replace("\n", " ")

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKD", text)

    def remove_html_tags(self, text: str) -> str:
        return re.sub(r"<.*?>", "", text)

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", "", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", "", text)

    def remove_hashtags_mentions(self, text: str) -> str:
        return re.sub(r"[@#]\S+", "", text)

    def remove_special_characters(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def remove_stopwords(self, text: str) -> str:
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def lemmatize(self, text: str) -> str:
        return " ".join([self.lemmatizer.lemmatize(w) for w in text.split()])

    def preprocess(self, text: str) -> str:
        text = self.to_lowercase(text)
        text = self.strip_whitespace(text)
        text = self.remove_extra_spaces(text)
        text = self.remove_tabs_and_newlines(text)
        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_hashtags_mentions(text)
        text = self.remove_special_characters(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["text"] = df["message"].apply(self.preprocess)
        df["message_length"] = df["message"].apply(len)
        df["word_count"] = df["message"].apply(lambda x: len(x.split()))
        df["punct_count"] = df["message"].apply(lambda x: sum(1 for c in x if c in "!?£$@#%&*"))
        df["digit_count"] = df["message"].apply(lambda x: sum(1 for c in x if c.isdigit()))
        df = df.drop(columns=["message"])
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        return df