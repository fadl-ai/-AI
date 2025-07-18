Fake News Detector Backend

This is the backend implementation of our Fake News Detector project. The backend uses a machine learning model to analyze text and determine if it's likely to be fake news.

Setup Instructions

1. Create a Python virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install required packages:
   pip install -r requirements.txt

3. Create required directories:
   mkdir logs
   mkdir models

4. Place the trained model files in the models/fake_news_model directory

Running the Backend

To run the backend server:
   python app.py

The server will start on http://localhost:5000

API Endpoint

The main endpoint is:
- POST /analyze
  - Accepts JSON with a "text" field containing the news article to analyze
  - Returns analysis results including fake/real prediction and confidence scores

Note
This backend requires a trained model to be placed in the models/fake_news_model directory. Without the model files, the application will not run.
