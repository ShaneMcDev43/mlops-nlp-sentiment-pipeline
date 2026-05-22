import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

model = load_model("sentiment_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded)[0][0]

    label = "positive" if prediction >= 0.5 else "negative"

    return {
        "text": text,
        "prediction": label,
        "confidence": float(prediction)
    }