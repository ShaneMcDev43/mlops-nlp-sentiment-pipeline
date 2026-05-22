# mlops-nlp-sentiment-pipeline

# NLP Sentiment Analysis MLOps Pipeline

This project demonstrates an end-to-end MLOps pipeline for sentiment analysis using a Bidirectional LSTM model.

## Features

- Text preprocessing
- Sentiment prediction API
- Flask deployment
- Docker containerisation
- GitHub Actions CI/CD
- Automated unit testing

## Technologies Used

- Python
- TensorFlow / Keras
- Flask
- Docker
- GitHub Actions

## API Endpoint

POST `/predict`

Example request:

```json
{
  "text": "I love this project"
}
```

Example response:

```json
{
  "text": "I love this project",
  "prediction": "positive",
  "confidence": 0.98
}
```

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask API:

```bash
python app/app.py
```

## Running Tests

```bash
pytest
```

## Docker

Build Docker image:

```bash
docker build -t mlops-nlp-api .
```

Run Docker container:

```bash
docker run -p 5000:5000 mlops-nlp-api
```