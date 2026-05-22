import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import clean_text


MAX_LEN = 50
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"


model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(text):
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

    prediction = model.predict(padded)[0][0]

    label = "positive" if prediction > 0.5 else "negative"

    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "prediction": label,
        "confidence": float(prediction)
    }