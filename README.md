# OTT Review Intelligence & Personalization System

A Streamlit application that analyzes movie reviews and provides personalized movie recommendations using machine learning.

## Features

- **Sentiment Analysis**: Analyzes movie reviews to determine positive/negative sentiment
- **Preference Scoring**: Calculates preference score based on sentiment probability
- **Keyword Extraction**: Extracts top keywords using TF-IDF scores
- **Personalized Recommendations**: Recommends movies based on cosine similarity between review and movie descriptions
- **Modern UI**: Clean, responsive interface with color-coded similarity indicators

## Files Required

- `app.py` - Main Streamlit application
- `sentiment_model.joblib` - Pre-trained sentiment analysis model
- `tfidf_vectorizer.joblib` - TF-IDF vectorizer for text processing
- `movie_tfidf_matrix.joblib` - Pre-computed TF-IDF matrix for movies
- `movies_metadata_cleaned.csv` - Movie metadata dataset
- `requirements.txt` - Python dependencies

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Navigate to the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the provided URL (usually `http://localhost:8501`)

## How It Works

1. **Text Processing**: User reviews are preprocessed using NLTK (lowercase, tokenization, stopword removal, lemmatization)
2. **Feature Extraction**: Reviews are transformed using TF-IDF vectorization
3. **Sentiment Analysis**: Pre-trained model predicts sentiment and calculates preference score
4. **Similarity Matching**: Cosine similarity between review and movie descriptions
5. **Recommendations**: Top 5 most similar movies are displayed with similarity scores

## Deployment Options

### 1. Streamlit Community Cloud
- Push your code to GitHub
- Connect to [Streamlit Community Cloud](https://share.streamlit.io/)
- Deploy with one click

### 2. Heroku
- Create `Procfile`:
  ```
  web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
  ```
- Deploy to Heroku

### 3. Railway
- Connect GitHub repository
- Railway will auto-detect and deploy the Streamlit app

### 4. AWS/GCP/Azure
- Use Docker containerization
- Deploy to cloud platform of choice

## Model Information

- **Sentiment Model**: Trained on movie review data
- **TF-IDF Vectorizer**: 5000 features, fitted on movie descriptions
- **Movie Dataset**: 5 movies with metadata and descriptions
- **Similarity Algorithm**: Cosine similarity between review and movie vectors

## Technical Stack

- **Backend**: Python, Streamlit
- **ML Libraries**: scikit-learn, NLTK, pandas, numpy
- **Frontend**: Streamlit components with custom CSS
- **Deployment**: Compatible with major cloud platforms

## Usage

1. Enter a movie review in the text area
2. Click "Analyze Review"
3. View:
   - Sentiment analysis result
   - Preference score (0-100%)
   - Top keywords extracted from review
   - Personalized movie recommendations with similarity scores

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.
