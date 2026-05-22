import pickle

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import prepare_data
from src.train import MAX_LEN


MODEL_PATH = "sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"


def evaluate_model(file_path):
    X_train, X_test, y_train, y_test = prepare_data(file_path)

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    loss, accuracy = model.evaluate(X_test_pad, y_test)

    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model("data/training.1600000.processed.noemoticon.csv")