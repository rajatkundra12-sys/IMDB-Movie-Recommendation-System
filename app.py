import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive design
def local_css():
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            font-family: 'Poppins', sans-serif;
        }
        .stTextArea textarea, .stTextInput input, textarea, input {
    caret-color: #000000 !important;
        }
        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0 2rem 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            text-align: center;
            animation: slideDown 0.8s ease;
        }
        
        @keyframes slideDown {
            from {
                transform: translateY(-50px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Movie Card */
        .movie-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            border-left: 5px solid #667eea;
            animation: fadeIn 0.5s ease;
        }
        
        .movie-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Rank Badge */
        .rank-badge {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1rem;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Stats Card */
        .stats-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
            color: white;
        }
        
        .stats-card:hover {
            transform: scale(1.05);
            background: rgba(255,255,255,0.2);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .stats-card h2 {
            font-size: 2rem;
            margin: 0.5rem 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.5);
        }
        
        /* Text Area */
        .stTextArea > div > div > textarea {
            border-radius: 15px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            font-size: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.95);
            transition: all 0.3s ease;
            color: #000000; 
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
            transform: scale(1.02);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #16213e, #1a1a2e);
        }
        
        .sidebar-content {
            padding: 2rem 1rem;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
        }
        
        /* Success Message */
        .success-message {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            animation: slideIn 0.5s ease;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(-100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Info Box */
        .info-box {
            background: rgba(255,255,255,0.95);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 3rem;
            box-shadow: 0 -5px 20px rgba(102, 126, 234, 0.3);
        }
        
        /* Star Rating */
        .stars {
            color: #FFD700;
            font-size: 1.2rem;
            letter-spacing: 3px;
            text-shadow: 0 0 10px rgba(255,215,0,0.3);
        }
        
        /* Feature Card */
        .feature-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(5px);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.2);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
        }
        
        .feature-card h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* Movie Grid Item */
        .movie-grid-item {
            background: rgba(255,255,255,0.95);
            padding: 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .movie-grid-item:hover {
            border-color: #667eea;
            transform: translateX(5px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        /* Example Button */
        .example-btn {
            background: rgba(255,255,255,0.1);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            margin: 0.2rem;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .example-btn:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Metric Card */
        .metric-card {
            background: linear-gradient(135deg, #667eea20, #764ba220);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #667eea40;
        }
        
        /* Divider */
        .custom-divider {
            height: 3px;
            background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
            margin: 2rem 0;
        }
        
        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 5px;
            font-size: 0.8rem;
            white-space: nowrap;
            z-index: 1000;
        }
    </style>
    """, unsafe_allow_html=True)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

# Initialize NLP components
@st.cache_resource
def load_nlp_components():
    """Load and cache NLP components"""
    download_nltk_data()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Load data
@st.cache_data
def load_data():
    """Load and cache the movie data"""
    try:
        df = pd.read_csv("imdb_150_movies_2024.csv")
        
        # Clean the data
        df = df[df["Movie Name"] != "Recently viewed"]

        df = df.reset_index(drop=True)
        return df
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        return create_sample_data()

def create_sample_data():
    """Create sample movie data for demonstration"""
    data = {
        'Movie Name': [
            'Fallout', 'Dune: Part Two', 'Deadpool 3', 'Kingdom of the Planet of the Apes',
            'Mad Max: Fury Road', 'The Batman 2', 'John Wick 5', 'Mission: Impossible 8',
            'Godzilla x Kong', 'Alien: Romulus', 'Oppenheimer', 'Barbie',
            'Killers of the Flower Moon', 'Poor Things', 'The Holdovers'
        ],
        'Storyline': [
            'In a future, post-apocalyptic Los Angeles brought by nuclear devastation, survivors must fight for existence in a harsh wasteland while uncovering dark secrets about the past.',
            'Paul Atreides unites with Chani and the Fremen while seeking revenge against the conspirators who destroyed his family, leading to an epic war for control of the spice melange.',
            'The time-traveling mercenary Deadpool joins forces with Wolverine to save the multiverse from destruction in this action-packed comedy adventure.',
            'Many years after the reign of Caesar, a young ape goes on a journey that will lead him to question everything he was taught about the past and the future of apes and humans.',
            'In a post-apocalyptic wasteland, Max joins forces with Imperator Furiosa to escape a tyrant and her warlords in this high-octane action thriller.',
            'Batman investigates a series of mysterious murders in Gotham City while facing new villains and uncovering dark secrets about his own family.',
            'John Wick continues his fight against the High Table, facing new challenges and old allies in the dangerous underworld of assassins.',
            'Ethan Hunt and his IMF team face their most dangerous mission yet, racing against time to prevent a global catastrophe involving rogue AI.',
            'The legendary Kong and the fearsome Godzilla must team up against a colossal undiscovered threat hidden deep within the Earth.',
            'Young space colonists encounter the most terrifying life form in the universe while exploring a derelict space station on a distant planet.',
            'The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II.',
            'Barbie and Ken discover the joys and challenges of living in the real world after leaving their perfect Barbie Land.',
            'When oil is discovered in 1920s Oklahoma under Osage Nation land, the Osage people are murdered one by one until the FBI steps in.',
            'The incredible tale about the fantastical evolution of Bella Baxter, a young woman brought back to life by the brilliant scientist Dr. Godwin Baxter.',
            'A cranky history teacher at a prestigious New England prep school is forced to remain on campus during Christmas break with a grieving student and the head cook.'
        ]
    }
    return pd.DataFrame(data)

# Create TF-IDF matrix
@st.cache_resource
def create_tfidf_matrix(_df):
    """Create and cache TF-IDF matrix"""
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    
    def remove_stopwords(text):
        words = text.split()
        filtered = [word for word in words if word not in stop_words]
        return " ".join(filtered)
    
    def lemmatize_text(text):
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    
    # Clean storylines
    cleaned = _df["Storyline"].apply(clean_text)
    cleaned = cleaned.apply(remove_stopwords)
    cleaned = cleaned.apply(lemmatize_text)
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(cleaned)
    
    return tfidf, tfidf_matrix, cleaned

# Load all components
stop_words, lemmatizer = load_nlp_components()
df = load_data()
tfidf, tfidf_matrix, cleaned_storylines = create_tfidf_matrix(df)

# Apply custom CSS
local_css()

# Header
st.markdown("""
<div class='main-header'>
    <h1 style='color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🎬 CineMatch Pro</h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 1.2rem; margin-top: 0.5rem;'>
        Discover Your Next Favorite Movie with AI-Powered Storyline Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 2rem;'>🎯 Dashboard</h2>", unsafe_allow_html=True)
    
    menu = st.radio(
        "",
        ["🏠 Home", "🎯 Recommendations", "ℹ️ About"],
        index=0
    )
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
 
    # Quick tip
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea20, #764ba220); padding: 1rem; border-radius: 12px; border: 1px solid #667eea40;'>
        <p style='color: white; margin: 0;'>
            💡 <strong>Pro Tip:</strong> Enter any storyline idea and get 5 similar movie recommendations instantly!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # RECOMMENDATION FUNCTION - YEH ADD KARO
def recommend_movies(user_input, top_n=5):
    # Clean input
    text = user_input.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_input = " ".join(words)
    
    # Transform and calculate similarity
    input_vector = tfidf.transform([cleaned_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    similarity_scores = list(enumerate(similarity_scores[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_movies = similarity_scores[:top_n]
    recommendations = []
    for idx, score in top_movies:
        recommendations.append({
            'Movie Name': df.iloc[idx]['Movie Name'],
            'Storyline': df.iloc[idx]['Storyline'],
            'Similarity': round(score * 100, 2)
        })
    
    return recommendations

# Home Page
if menu == "🏠 Home":
    # Welcome section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <h2 style='color: #667eea; margin-top: 0;'>Welcome to CineMatch Pro! 🎬</h2>
            <p style='color: #444; font-size: 1.1rem; line-height: 1.6;'>
                Our intelligent recommendation system helps you analyze movies by storylines 
                and find the perfect matches for your interests. Simply describe what kind of story you're looking for!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        st.markdown("<h3 style='color: white; margin: 1.5rem 0 1rem 0;'>✨ Key Features</h3>", unsafe_allow_html=True)
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.markdown("""
            <div class='feature-card'>
                <h1>🎯</h1>
                <h4>Smart Matching</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div class='feature-card'>
                <h1>⚡</h1>
                <h4>Real-time movies</h4>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='color: white; text-align: center; margin-bottom: 1rem;'>🔥 Featured Movies</h3>", unsafe_allow_html=True)
        
        # Show random movies
        sample_movies = df.sample(min(5, len(df)))
        for _, movie in sample_movies.iterrows():
            st.markdown(f"""
            <div class='movie-grid-item'>
                <h4 style='color: #667eea; margin: 0;'>{movie['Movie Name']}</h4>
                <p style='color: #666; font-size: 0.85rem; margin: 0.3rem 0 0 0;'>{movie['Storyline'][:70]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
# Recommendations Page
elif menu == "🎯 Recommendations":
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>🎯 Get Personalized Recommendations</h2>", unsafe_allow_html=True)
    
    # Simple columns for centering
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        user_input = st.text_area(
            "✨ Enter your storyline idea:",
            height=120,
            placeholder="E.g., A young hero discovers hidden powers...",
            key="story_input"
        )
        
        recommend_button = st.button("🔍 Find Similar Movies", use_container_width=True)
    
    # Recommendations display
    if recommend_button and user_input:
        with st.spinner("🎬 Finding similar movies..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            recommendations = recommend_movies(user_input)  # Ye function call kar raha hai
            progress_bar.empty()
        
        # Show results
        for i, rec in enumerate(recommendations, 1):
            stars = "⭐" * min(5, int(rec['Similarity'] // 20 + 1))
            
            st.markdown(f"""
            <div class='movie-card'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <div class='rank-badge'>#{i}</div>
                    <h2 style='color: #667eea; margin: 0;'>{rec['Movie Name']}</h2>
                </div>
                <div><p style='color: #444;'>{rec['Storyline'][:150]}...</p>
                </div>
                
            </div>
            """, unsafe_allow_html=True)
# About Page
else:  
    st.markdown("<h2 class='gradient-text'>ℹ️ About The Project</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This IMDb Movie Recommendation System helps you to recommend movies based on storyline similarity. Users can input a movie storyline or idea, and the system will suggest the top 5 most similar movies from our system.
        """)
# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p style='margin: 0; font-size: 1.2rem;'>🎬 CineMatch Pro</p>
    <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>© IMDb Movie Recommendation System</p>
    <p style='margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.8;'>Made with ❤️ using Python & Streamlit</p>
</div>
""", unsafe_allow_html=True)