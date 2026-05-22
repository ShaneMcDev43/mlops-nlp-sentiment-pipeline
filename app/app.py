from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "NLP Sentiment API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    from src.predict import predict_sentiment

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide text input"}), 400

    result = predict_sentiment(data["text"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)