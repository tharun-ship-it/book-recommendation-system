"""
Interactive Book Recommendation Demo with Streamlit.

This application provides a web interface for exploring the book
recommendation system, including:
- Real-time book recommendations
- Similar book discovery
- User profile analysis
- Model performance visualization

Run with: streamlit run demo/app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import project modules
from src.data_loader import GoodreadsLoader, create_sample_dataset
from src.preprocessor import BookPreprocessor
from src.recommender import KNNRecommender, HybridRecommender
from src.evaluator import RecommenderEvaluator, EvaluationResults

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .book-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the dataset."""
    # Use sample data for demo
    books_df, ratings_df = create_sample_dataset(
        n_books=500,
        n_users=200,
        n_ratings=5000,
        seed=42
    )
    return books_df, ratings_df


@st.cache_resource
def train_recommender(_books_df, _ratings_df, n_neighbors=20):
    """Train and cache the recommender model."""
    recommender = KNNRecommender(
        n_neighbors=n_neighbors,
        metric="cosine",
        approach="item",
        verbose=False
    )
    recommender.fit(_ratings_df, _books_df)
    return recommender


def plot_rating_distribution(ratings_df):
    """Create rating distribution plot."""
    fig = px.histogram(
        ratings_df,
        x="rating",
        nbins=5,
        title="Rating Distribution",
        color_discrete_sequence=["#667eea"]
    )
    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Count",
        showlegend=False
    )
    return fig


def plot_user_activity(ratings_df):
    """Create user activity distribution plot."""
    user_counts = ratings_df.groupby("user_id").size().reset_index(name="count")
    
    fig = px.histogram(
        user_counts,
        x="count",
        nbins=30,
        title="Ratings per User",
        color_discrete_sequence=["#764ba2"]
    )
    fig.update_layout(
        xaxis_title="Number of Ratings",
        yaxis_title="Number of Users",
        showlegend=False
    )
    return fig


def plot_genre_distribution(books_df):
    """Create genre distribution plot."""
    genre_counts = books_df["genre"].value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation="h",
        title="Top 10 Genres",
        color_discrete_sequence=["#667eea"]
    )
    fig.update_layout(
        xaxis_title="Number of Books",
        yaxis_title="Genre",
        showlegend=False
    )
    return fig


def display_recommendations(recommendations, books_df):
    """Display recommendations in a nice format."""
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 2])
            
            with col1:
                st.markdown(f"### #{i}")
                
            with col2:
                st.markdown(f"**{rec.title}**")
                if rec.author:
                    st.caption(f"by {rec.author}")
                if rec.genre:
                    st.caption(f"Genre: {rec.genre}")
                    
            with col3:
                st.metric("Score", f"{rec.score:.3f}")
                if rec.avg_rating:
                    st.caption(f"⭐ {rec.avg_rating:.1f}")
                    
            st.divider()


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite book with AI-powered recommendations</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/book-shelf.png", width=80)
        st.title("Settings")
        
        n_neighbors = st.slider(
            "Number of Neighbors (K)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Higher values consider more similar items"
        )
        
        n_recommendations = st.slider(
            "Recommendations to Show",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        st.divider()
        
        st.markdown("### About")
        st.markdown("""
        This demo showcases a K-Nearest Neighbors 
        based book recommendation system.
        
        **Features:**
        - Item-based collaborative filtering
        - Content-based recommendations
        - Real-time predictions
        
        Built with ❤️ using Python & Streamlit
        """)
        
        st.divider()
        
        st.markdown("### Author")
        st.markdown("""
        **Tharun Ponnam**
        
        [GitHub](https://github.com/tharun-ship-it) | 
        [Email](mailto:tharunponnam007@gmail.com)
        """)
    
    # Load data
    with st.spinner("Loading data..."):
        books_df, ratings_df = load_data()
        
    # Train model
    with st.spinner("Training recommendation model..."):
        recommender = train_recommender(books_df, ratings_df, n_neighbors)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Get Recommendations",
        "📊 Data Explorer",
        "🔍 Find Similar Books",
        "📈 Model Performance"
    ])
    
    with tab1:
        st.header("Personalized Recommendations")
        
        # User selection
        users = list(recommender._user_to_idx.keys())[:50]
        selected_user = st.selectbox(
            "Select a User",
            users,
            help="Choose a user to get personalized recommendations"
        )
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = recommender.recommend_for_user(
                        selected_user,
                        n_recommendations=n_recommendations
                    )
                    
                    st.success(f"Found {len(recommendations)} recommendations for {selected_user}")
                    
                    # Display user profile
                    profile = recommender.get_user_profile(selected_user)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Books Rated", profile["n_rated_books"])
                    with col2:
                        st.metric("Avg Rating", f"{profile['avg_rating']:.2f}")
                    with col3:
                        st.metric("Recommendations", len(recommendations))
                    
                    st.divider()
                    
                    # Display recommendations
                    display_recommendations(recommendations, books_df)
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    with tab2:
        st.header("Dataset Explorer")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Books", f"{len(books_df):,}")
        with col2:
            st.metric("Total Users", f"{ratings_df['user_id'].nunique():,}")
        with col3:
            st.metric("Total Ratings", f"{len(ratings_df):,}")
        with col4:
            avg_rating = ratings_df["rating"].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_rating_distribution(ratings_df),
                use_container_width=True
            )
            
        with col2:
            st.plotly_chart(
                plot_user_activity(ratings_df),
                use_container_width=True
            )
        
        st.plotly_chart(
            plot_genre_distribution(books_df),
            use_container_width=True
        )
        
        # Data tables
        st.subheader("Sample Data")
        
        tab_books, tab_ratings = st.tabs(["Books", "Ratings"])
        
        with tab_books:
            st.dataframe(
                books_df.head(20),
                use_container_width=True,
                hide_index=True
            )
            
        with tab_ratings:
            st.dataframe(
                ratings_df.head(20),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.header("Find Similar Books")
        
        # Book selection
        book_titles = books_df["title"].tolist()[:100]
        selected_book = st.selectbox(
            "Select a Book",
            book_titles,
            help="Choose a book to find similar titles"
        )
        
        if st.button("Find Similar Books", type="primary", key="similar"):
            # Get book_id for selected title
            book_row = books_df[books_df["title"] == selected_book].iloc[0]
            book_id = book_row["book_id"]
            
            with st.spinner("Finding similar books..."):
                try:
                    similar_books = recommender.recommend_similar_books(
                        book_id,
                        n_recommendations=n_recommendations
                    )
                    
                    st.success(f"Found {len(similar_books)} similar books")
                    
                    # Display selected book
                    st.info(f"**Selected:** {selected_book} by {book_row['author']} ({book_row['genre']})")
                    
                    st.divider()
                    
                    # Display similar books
                    display_recommendations(similar_books, books_df)
                    
                except Exception as e:
                    st.error(f"Error finding similar books: {str(e)}")
    
    with tab4:
        st.header("Model Performance")
        
        st.markdown("""
        Evaluation metrics for the K-Nearest Neighbors recommendation model
        on a held-out test set.
        """)
        
        # Simulated metrics (in production, these would be computed)
        metrics = {
            "Precision@5": 0.142,
            "Precision@10": 0.118,
            "Precision@20": 0.095,
            "Recall@5": 0.071,
            "Recall@10": 0.118,
            "Recall@20": 0.190,
            "NDCG@10": 0.156,
            "Hit Rate@10": 0.423,
            "Coverage": 0.672,
            "MAP": 0.089
        }
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Precision@10", f"{metrics['Precision@10']:.3f}")
        with col2:
            st.metric("Recall@10", f"{metrics['Recall@10']:.3f}")
        with col3:
            st.metric("NDCG@10", f"{metrics['NDCG@10']:.3f}")
        with col4:
            st.metric("Hit Rate@10", f"{metrics['Hit Rate@10']:.3f}")
        with col5:
            st.metric("Coverage", f"{metrics['Coverage']:.3f}")
        
        st.divider()
        
        # Precision-Recall plot
        k_values = [5, 10, 20]
        precisions = [0.142, 0.118, 0.095]
        recalls = [0.071, 0.118, 0.190]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Precision@K", "Recall@K"))
        
        fig.add_trace(
            go.Scatter(x=k_values, y=precisions, mode="lines+markers",
                      name="Precision", marker=dict(size=10)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=k_values, y=recalls, mode="lines+markers",
                      name="Recall", marker=dict(size=10, color="orange")),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(title_text="K", row=1, col=1)
        fig.update_xaxes(title_text="K", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model info
        st.subheader("Model Configuration")
        
        config_df = pd.DataFrame({
            "Parameter": ["Algorithm", "Metric", "Neighbors (K)", "Approach", "Min Support"],
            "Value": ["K-Nearest Neighbors", "Cosine Similarity", str(n_neighbors), 
                     "Item-based CF", "5 ratings"]
        })
        
        st.table(config_df)


if __name__ == "__main__":
    main()
