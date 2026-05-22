import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocess import prepare_data


MAX_WORDS = 20000
MAX_LEN = 50
MODEL_PATH = "sentiment_model.keras"
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


def train_model(file_path):
    X_train, X_test, y_train, y_test = prepare_data(file_path)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    model = build_bilstm_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train_pad,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    loss, accuracy = model.evaluate(X_test_pad, y_test)

    print("BiLSTM Test Loss:", loss)
    print("BiLSTM Test Accuracy:", accuracy)

    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    train_model("data/training.1600000.processed.noemoticon.csv")