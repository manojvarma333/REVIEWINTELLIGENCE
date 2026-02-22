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
from typing import List, Tuple, Dict

# Download NLTK resources (only needed once)
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Load cached models and data
@st.cache_resource
def load_models_and_data():
    """Load all pre-trained models and data files"""
    sentiment_model = joblib.load('sentiment_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    movies_df = pd.read_csv('movies_metadata_cleaned.csv')
    movie_tfidf_matrix = joblib.load('movie_tfidf_matrix.joblib')
    
    return sentiment_model, tfidf_vectorizer, movies_df, movie_tfidf_matrix

# Text preprocessing functions
@st.cache_data
def get_preprocessing_objects():
    """Get stopwords and lemmatizer objects"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Preprocess text with the same logic as training:
    - Convert to lowercase
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
    
    return ' '.join(processed_tokens)

def extract_top_keywords(tfidf_vector: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[str]:
    """Extract top keywords based on TF-IDF scores"""
    # Get all scores and indices
    scores = tfidf_vector.toarray().flatten()
    
    # Get all indices sorted by score (including zeros)
    sorted_indices = np.argsort(scores)[::-1]
    
    # Get top k keywords (even if score is zero, we want the highest scoring terms)
    top_keywords = []
    for i in sorted_indices[:top_k]:
        if scores[i] > 0:  # Only include if score > 0
            top_keywords.append(feature_names[i])
        else:
            # If no more positive scores, break
            break
    
    # If no keywords found with positive scores, get highest scoring anyway
    if not top_keywords:
        for i in sorted_indices[:top_k]:
            top_keywords.append(feature_names[i])
    
    return top_keywords

def get_movie_recommendations(user_tfidf_vector, movie_tfidf_matrix, movies_df, top_k: int = 5) -> List[Dict]:
    """Get top movie recommendations based on cosine similarity"""
    # Compute cosine similarity
    similarities = cosine_similarity(user_tfidf_vector, movie_tfidf_matrix).flatten()
    
    # Get top indices sorted by similarity (highest first)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create recommendation list
    recommendations = []
    for idx in top_indices:
        if idx < len(movies_df):
            movie = movies_df.iloc[idx]
            recommendations.append({
                'title': movie['movie_title'],
                'genre': movie['genre'],
                'description': movie['description'],
                'similarity_score': similarities[idx]
            })
    
    return recommendations

def main():
    # Page configuration
    st.set_page_config(
        page_title="OTT Review Intelligence & Personalization System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load models and data
    sentiment_model, tfidf_vectorizer, movies_df, movie_tfidf_matrix = load_models_and_data()
    stop_words, lemmatizer = get_preprocessing_objects()
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .sentiment-positive {
            color: #28a745;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .sentiment-negative {
            color: #dc3545;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .sentiment-neutral {
            color: #6c757d;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .movie-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        .keyword-chip {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem;
            border-radius: 15px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .preference-score {
            font-size: 1.1rem;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 OTT Review Intelligence & Personalization System</h1>
        <p>Analyze movie reviews and get personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Review")
        
        # Text input for review
        user_review = st.text_area(
            "Write your movie review here:",
            placeholder="Enter your detailed movie review...",
            height=150
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
        
        if analyze_button and user_review.strip():
            with st.spinner("Analyzing your review..."):
                try:
                    # Preprocess the review
                    processed_review = preprocess_text(user_review, stop_words, lemmatizer)
                    
                    # Transform using TF-IDF
                    review_tfidf = tfidf_vectorizer.transform([processed_review])
                    
                    # Predict sentiment
                    sentiment_prediction = sentiment_model.predict(review_tfidf)[0]
                    sentiment_proba = sentiment_model.predict_proba(review_tfidf)[0]
                    
                    # Get preference score (probability of positive sentiment)
                    if len(sentiment_model.classes_) > 1 and sentiment_model.classes_[1] == 1:
                        preference_score = sentiment_proba[1]
                    else:
                        preference_score = sentiment_proba[0] if len(sentiment_proba) > 0 else 0.5
                    
                    # Extract top keywords
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    top_keywords = extract_top_keywords(review_tfidf, feature_names, top_k=5)
                    
                    # Get movie recommendations
                    recommendations = get_movie_recommendations(
                        review_tfidf, movie_tfidf_matrix, movies_df, top_k=5
                    )
                    
                    # Store results in session state
                    st.session_state.analysis_results = {
                        'sentiment': sentiment_prediction,
                        'preference_score': preference_score,
                        'keywords': top_keywords,
                        'recommendations': recommendations,
                        'original_review': user_review,
                        'processed_review': processed_review
                    }
                    
                    # Debug info (remove later)
                    st.write(f"🔍 Debug: Processed review: '{processed_review}'")
                    st.write(f"🔍 Debug: Found {len(top_keywords)} keywords: {top_keywords}")
                    st.write(f"🔍 Debug: Sentiment classes: {sentiment_model.classes_}")
                    st.write(f"🔍 Debug: Sentiment probabilities: {sentiment_proba}")
                    
                    # Show similarity scores for all movies
                    st.write("🔍 Debug: Movie similarity scores:")
                    similarities = cosine_similarity(review_tfidf, movie_tfidf_matrix).flatten()
                    for i, (idx, movie) in enumerate(zip(range(len(movies_df)), movies_df.iterrows())):
                        movie_title = movie[1]['movie_title']
                        similarity = similarities[i]
                        st.write(f"  {movie_title}: {similarity:.6f}")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"Review length: {len(user_review)}")
                    st.write(f"Processed review: {processed_review if 'processed_review' in locals() else 'Not processed'}")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        # Display results if available
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Sentiment display
            sentiment = results['sentiment']
            if sentiment == 1:
                sentiment_text = "😊 Positive"
                sentiment_class = "sentiment-positive"
            elif sentiment == 0:
                sentiment_text = "😔 Negative"
                sentiment_class = "sentiment-negative"
            else:
                sentiment_text = "😐 Neutral"
                sentiment_class = "sentiment-neutral"
            
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <h4>Sentiment Analysis</h4>
                <p class="{sentiment_class}">{sentiment_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preference score
            preference_score = results['preference_score']
            st.markdown("**Preference Score (Positive Sentiment Probability):**")
            st.progress(preference_score)
            st.markdown(f"<p class='preference-score'>{preference_score:.2%}</p>", unsafe_allow_html=True)
            
            # Keywords
            keywords = results['keywords']
            if keywords:
                st.markdown("**🔑 Top Keywords:**")
                keywords_html = " ".join([f'<span class="keyword-chip">{keyword}</span>' for keyword in keywords])
                st.markdown(keywords_html, unsafe_allow_html=True)
            else:
                st.markdown("**🔑 Top Keywords:** No significant keywords found")
    
    # Movie Recommendations Section
    if 'analysis_results' in st.session_state:
        st.markdown("---")
        st.subheader("🎬 Personalized Movie Recommendations")
        
        recommendations = st.session_state.analysis_results['recommendations']
        
        if recommendations:
            st.write(f"🎯 Found {len(recommendations)} movie recommendations for you!")
            
            # Create columns for recommendations
            cols = st.columns(len(recommendations))
            
            for i, (col, movie) in enumerate(zip(cols, recommendations)):
                with col:
                    # Color code based on similarity score
                    if movie['similarity_score'] > 0.05:
                        card_color = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"  # Green for high similarity
                        emoji = "🔥"
                    elif movie['similarity_score'] > 0.02:
                        card_color = "linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)"  # Orange for medium similarity
                        emoji = "⭐"
                    else:
                        card_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"  # Purple for low similarity
                        emoji = "🎬"
                    
                    st.markdown(f"""
                    <div class="movie-card" style="background: {card_color}; color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                        <h4 style="color: white; margin-bottom: 1rem;">{emoji} {movie['title']}</h4>
                        <p style="color: white; margin-bottom: 0.5rem;"><strong>🎭 Genre:</strong> {movie['genre']}</p>
                        <p style="color: white; margin-bottom: 0.5rem;"><strong>📊 Similarity:</strong> {movie['similarity_score']:.4f} ({movie['similarity_score']:.2%})</p>
                        <p style="color: white; font-size: 0.9rem; line-height: 1.4;">{movie['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No recommendations available. Please try with a different review.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
