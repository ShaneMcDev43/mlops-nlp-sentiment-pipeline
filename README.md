# mlops-nlp-sentiment-pipeline

# NLP Sentiment Analysis MLOps Pipeline

This project demonstrates an end-to-end MLOps pipeline for sentiment analysis using a Bidirectional LSTM model.

## Model Artefact Note

The trained `.keras` model file is not stored directly in this GitHub repository because it exceeds GitHub's standard upload size limit. The model is generated using `src/train.py` and can be downloaded or transferred to the deployment server separately.

This reflects a common MLOps practice where large model artefacts are stored separately from source code.

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

## Deployment Status

The Flask API was successfully deployed on a Linux VM using Docker.

Example prediction:

```json
{
  "cleaned_text": "i love this project",
  "confidence": 0.879081666469574,
  "prediction": "positive",
  "text": "I love this project"
}