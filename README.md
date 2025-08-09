# Restaurant Review Sentiment API

A lightweight REST API for sentiment analysis focused on **restaurant reviews**, built with **FastAPI** and a pre-trained machine learning model. The API classifies customer feedback as **positive**, **negative**, or **neutral**, helping restaurants gain insights from customer opinions.

## 🚀 Features

- ⚡ High-performance API with FastAPI
- 🍽️ Tailored for analyzing restaurant review sentiment
- 🧠 Pre-trained ML model for accurate predictions
- 📄 Interactive Swagger docs available at `/docs`

## 🧰 Tech Stack

- Python 3.10+
- FastAPI
- transformers
- torch
- Uvicorn

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-api.git
cd sentiment-api
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/macOS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Running the API

From the root of the project:

```bash
uvicorn app.main:app --reload
```

Then go to: http://127.0.0.1:8000/docs to test the API via Swagger UI.


## 📬 Example Request

```bash
POST /predict
Content-Type: application/json

{
  "texto": "Buena comida , cena con los compañeros de trabajo, buen menú y precio, la comida espectacular, las puntillas geniales."
}
```

Response:
```json
{
    "texto": "Buena comida , cena con los compañeros de trabajo, buen menú y precio, la comida espectacular, las puntillas geniales.",
    "positividad": 0.9409984946250916,
    "negatividad": 0.012604037299752235,
    "neutralidad": 0.04639745131134987,
    "rating": 4.86,
    "fecha": "2025-07-16T18:26:04.945263",
    "length": 118
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

Feel free to use, modify, and distribute it for personal or commercial purposes, as long as the original license is included.
