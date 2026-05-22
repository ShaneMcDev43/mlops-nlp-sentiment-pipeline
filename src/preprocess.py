import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(file_path, sample_size_per_class=50000):
    column_names = ["target", "ids", "date", "flag", "user", "text"]

    df = pd.read_csv(
        file_path,
        encoding="latin-1",
        header=None,
        names=column_names
    )

    df = df[["target", "text"]]
    df["target"] = df["target"].replace(4, 1)

    negative_df = df[df["target"] == 0].sample(sample_size_per_class, random_state=42)
    positive_df = df[df["target"] == 1].sample(sample_size_per_class, random_state=42)

    df = pd.concat([negative_df, positive_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    df["clean_text"] = df["text"].apply(clean_text)

    df = df[df["clean_text"].str.strip() != ""]
    df = df.dropna(subset=["clean_text"])

    X = df["clean_text"].values
    y = df["target"].values

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )