import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import sys
import os

# Simple NLTK download
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        sentiment_model = joblib.load('sentiment_model.joblib')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        movies_df = pd.read_csv('movies_metadata_cleaned.csv')
        movie_tfidf_matrix = joblib.load('movie_tfidf_matrix.joblib')
        return sentiment_model, tfidf_vectorizer, movies_df, movie_tfidf_matrix
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Preprocessing
def preprocess_text(text):
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        
        processed_tokens = []
        for token in tokens:
            if token not in stop_words and len(token) > 2:
                lemmatized = lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return text

# Main app
def main():
    st.set_page_config(
        page_title="OTT Review Intelligence",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 OTT Review Intelligence System")
    st.write("Analyze movie reviews and get personalized recommendations")
    
    # Load models
    sentiment_model, tfidf_vectorizer, movies_df, movie_tfidf_matrix = load_models()
    
    if sentiment_model is None:
        st.error("❌ Failed to load models. Please check file paths.")
        return
    
    # Input
    user_review = st.text_area("Enter your movie review:", height=150)
    
    if st.button("🔍 Analyze Review"):
        if user_review.strip():
            try:
                with st.spinner("Processing..."):
                    # Preprocess
                    processed_review = preprocess_text(user_review)
                    
                    # Transform
                    user_vector = tfidf_vectorizer.transform([processed_review])
                    
                    # Predict sentiment
                    sentiment = sentiment_model.predict(user_vector)[0]
                    proba = sentiment_model.predict_proba(user_vector)[0]
                    
                    # Get preference score
                    if len(sentiment_model.classes_) > 1:
                        pref_score = proba[1] if sentiment_model.classes_[1] == 1 else proba[0]
                    else:
                        pref_score = proba[0] if len(proba) > 0 else 0.5
                    
                    # Get recommendations
                    similarities = cosine_similarity(user_vector, movie_tfidf_matrix).flatten()
                    top_indices = similarities.argsort()[-5:][::-1]
                    
                    recommendations = []
                    for idx in top_indices:
                        if idx < len(movies_df):
                            movie = movies_df.iloc[idx]
                            recommendations.append({
                                'title': movie['movie_title'],
                                'genre': movie['genre'],
                                'similarity': similarities[idx]
                            })
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Analysis Results")
                        
                        # Sentiment
                        if sentiment == 1:
                            st.success("😊 Positive Sentiment")
                        else:
                            st.error("😔 Negative Sentiment")
                        
                        # Preference score
                        st.write(f"**Preference Score:** {pref_score:.2%}")
                        st.progress(pref_score)
                        
                        # Keywords
                        feature_names = tfidf_vectorizer.get_feature_names_out()
                        scores = user_vector.toarray().flatten()
                        top_keywords_idx = np.argsort(scores)[-5:][::-1]
                        keywords = [feature_names[i] for i in top_keywords_idx if scores[i] > 0]
                        
                        if keywords:
                            st.write("**Keywords:** " + ", ".join(keywords))
                    
                    with col2:
                        st.subheader("🎬 Recommendations")
                        
                        for i, rec in enumerate(recommendations):
                            st.write(f"**{i+1}. {rec['title']}**")
                            st.write(f"*Genre:* {rec['genre']}")
                            st.write(f"*Similarity:* {rec['similarity']:.4f}")
                            st.write(f"*Description:* {rec['description']}")
                            st.write("---")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.write("Please try again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page.")
