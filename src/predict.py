import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import clean_text


MAX_WORDS = 20000
MAX_LEN = 50
WEIGHTS_PATH = "sentiment_weights.weights.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"


def build_bilstm_model():
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


model = build_bilstm_model()

# Build model before loading weights
model.build(input_shape=(None, MAX_LEN))

model.load_weights(WEIGHTS_PATH)

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