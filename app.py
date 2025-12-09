"""
SF Events Explorer - ML-Powered Event Discovery
================================================
Uses TF-IDF + feature extraction to match natural language queries to events.

Model: TfidfVectorizer trained on event corpus
Features: Age group, time of day, cost, category extraction from queries
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
from pathlib import Path

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SF Events Explorer",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0;
    }
    .event-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    .event-title {
        color: #f1f5f9;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .event-meta {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .category-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .cat-sports { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .cat-arts { background: rgba(139, 92, 246, 0.15); color: #a78bfa; }
    .cat-education { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .cat-family { background: rgba(244, 63, 94, 0.15); color: #fb7185; }
    .cat-health { background: rgba(6, 182, 212, 0.15); color: #22d3ee; }
    .free-tag { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .paid-tag { background: rgba(100, 116, 139, 0.15); color: #94a3b8; }
    .match-score {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ML MODEL CLASS
# ============================================================================
class EventSearchModel:
    """
    TF-IDF based event search model with feature extraction.
    
    Training:
        - Fits TfidfVectorizer on event corpus (name + description + category)
        - Learns vocabulary and IDF weights from 2,075 SF events
    
    Inference:
        - Transforms user query to TF-IDF vector
        - Computes cosine similarity with all events
        - Applies feature-based score boosting
        - Returns ranked results
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,           # Ignore very rare terms
            max_df=0.95         # Ignore very common terms
        )
        self.tfidf_matrix = None
        self.events_df = None
        self.is_trained = False
        
    def train(self, df: pd.DataFrame):
        """
        Train the TF-IDF model on event corpus.
        
        Args:
            df: DataFrame with event data
        """
        self.events_df = df.copy()
        
        # Create combined text field for each event
        self.events_df['search_text'] = (
            self.events_df['event_name'].fillna('') + ' ' +
            self.events_df['event_description'].fillna('') + ' ' +
            self.events_df['category'].fillna('') + ' ' +
            self.events_df['subcategory'].fillna('') + ' ' +
            self.events_df['age_group_eligibility_tags'].fillna('')
        )
        
        # Fit and transform - THIS IS THE TRAINING STEP
        self.tfidf_matrix = self.vectorizer.fit_transform(self.events_df['search_text'])
        self.is_trained = True
        
        return {
            'n_events': len(df),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'n_features': self.tfidf_matrix.shape[1]
        }
    
    def extract_query_features(self, query: str) -> dict:
        """
        Extract structured features from natural language query.
        
        This is rule-based feature extraction that augments the ML model.
        """
        query_lower = query.lower()
        features = {
            'age_group': None,
            'time_of_day': None,
            'is_free': None,
            'category': None,
            'is_weekend': None
        }
        
        # Age group detection
        age_patterns = {
            'kids': r'\b(kid|kids|child|children|toddler|preschool|pre-school)\b',
            'teens': r'\b(teen|teens|teenager|youth|young adult)\b',
            'families': r'\b(family|families|all ages)\b',
            'adults': r'\b(adult|adults|senior|seniors|21\+|over 21)\b',
            'babies': r'\b(baby|babies|infant|infants|newborn)\b'
        }
        for age_group, pattern in age_patterns.items():
            if re.search(pattern, query_lower):
                features['age_group'] = age_group
                break
        
        # Time of day detection
        time_patterns = {
            'morning': r'\b(morning|am|early|breakfast)\b',
            'afternoon': r'\b(afternoon|midday|lunch|noon)\b',
            'evening': r'\b(evening|night|pm|dinner|late)\b'
        }
        for time, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                features['time_of_day'] = time
                break
        
        # Weekend detection
        if re.search(r'\b(weekend|saturday|sunday|sat|sun)\b', query_lower):
            features['is_weekend'] = True
        
        # Free event detection
        if re.search(r'\b(free|no cost|complimentary)\b', query_lower):
            features['is_free'] = True
        
        # Category detection
        category_patterns = {
            'Sports & Recreation': r'\b(sport|sports|fitness|exercise|swim|basketball|soccer|tennis|pickleball|yoga|outdoor|hike|hiking|camping)\b',
            'Arts, Culture & Identity': r'\b(art|arts|music|dance|theater|theatre|film|photo|photography|painting|drawing|craft|creative)\b',
            'Education': r'\b(education|learn|learning|class|classes|workshop|stem|science|coding|programming|reading|library|book)\b',
            'Family Support': r'\b(family support|parenting|childcare|daycare)\b',
            'Health & Wellness': r'\b(health|wellness|meditation|mental health|therapy)\b'
        }
        for category, pattern in category_patterns.items():
            if re.search(pattern, query_lower):
                features['category'] = category
                break
        
        return features
    
    def get_time_of_day(self, time_str: str) -> str:
        """Convert time string to time of day."""
        if pd.isna(time_str) or not time_str:
            return 'unknown'
        try:
            hour = int(str(time_str).split(':')[0])
            if hour < 12:
                return 'morning'
            elif hour < 17:
                return 'afternoon'
            else:
                return 'evening'
        except:
            return 'unknown'
    
    def search(self, query: str, top_k: int = 12) -> pd.DataFrame:
        """
        Search for events matching the query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            
        Returns:
            DataFrame of matching events with scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get TF-IDF similarity scores
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Extract query features
        features = self.extract_query_features(query)
        
        # Calculate feature-based score boosts
        boosts = np.zeros(len(self.events_df))
        
        # Age group boost
        if features['age_group']:
            age_map = {
                'kids': ['Children', 'Pre-Teens', 'Toddlers', 'Pre-School'],
                'teens': ['Teens', 'Pre-Teens', 'TAY'],
                'families': ['Families', 'All Ages', 'Children'],
                'adults': ['TAY', 'Adults'],
                'babies': ['Infants', 'Toddlers', 'Babies']
            }
            target_ages = age_map.get(features['age_group'], [])
            for i, row in self.events_df.iterrows():
                age_tags = str(row.get('age_group_eligibility_tags', '')).lower()
                if any(age.lower() in age_tags for age in target_ages):
                    boosts[i] += 0.15
        
        # Time of day boost
        if features['time_of_day']:
            for i, row in self.events_df.iterrows():
                event_time = self.get_time_of_day(row.get('start_time', ''))
                if event_time == features['time_of_day']:
                    boosts[i] += 0.1
        
        # Free event boost
        if features['is_free']:
            for i, row in self.events_df.iterrows():
                if str(row.get('fee', '')).lower() != 'true':
                    boosts[i] += 0.15
        
        # Weekend boost
        if features['is_weekend']:
            for i, row in self.events_df.iterrows():
                days = str(row.get('days_of_week', '')).lower()
                if 'sa' in days or 'su' in days or 'sat' in days or 'sun' in days:
                    boosts[i] += 0.1
        
        # Category boost
        if features['category']:
            for i, row in self.events_df.iterrows():
                if features['category'] in str(row.get('category', '')):
                    boosts[i] += 0.2
        
        # Combine scores (TF-IDF similarity + feature boosts)
        final_scores = similarities + boosts
        
        # Get top results
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = self.events_df.iloc[top_indices].copy()
        results['match_score'] = final_scores[top_indices]
        results['tfidf_score'] = similarities[top_indices]
        results['feature_boost'] = boosts[top_indices]
        
        # Filter out zero-score results
        results = results[results['match_score'] > 0.01]
        
        return results
    
    def get_model_stats(self) -> dict:
        """Get statistics about the trained model."""
        if not self.is_trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'n_events': len(self.events_df),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'top_terms': self.get_top_terms(20)
        }
    
    def get_top_terms(self, n: int = 20) -> list:
        """Get the most important terms in the vocabulary."""
        if not self.is_trained:
            return []
        
        # Get feature names and their IDF scores
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        
        # Sort by IDF (lower = more common across docs)
        indices = np.argsort(idf_scores)[:n]
        return [feature_names[i] for i in indices]


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load and cache the events dataset."""
    df = pd.read_csv('data/events.csv')
    return df


@st.cache_resource
def get_trained_model(_df):
    """Train and cache the ML model."""
    model = EventSearchModel()
    stats = model.train(_df)
    return model, stats


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_time(time_str):
    """Format time string to 12-hour format."""
    if pd.isna(time_str) or not time_str:
        return "TBD"
    try:
        parts = str(time_str).split(':')
        hour = int(parts[0])
        minute = parts[1] if len(parts) > 1 else '00'
        ampm = 'AM' if hour < 12 else 'PM'
        hour_12 = hour % 12 or 12
        return f"{hour_12}:{minute} {ampm}"
    except:
        return str(time_str)


def get_category_class(category):
    """Get CSS class for category tag."""
    if pd.isna(category):
        return 'cat-sports'
    cat_lower = str(category).lower()
    if 'sport' in cat_lower:
        return 'cat-sports'
    elif 'art' in cat_lower or 'culture' in cat_lower:
        return 'cat-arts'
    elif 'education' in cat_lower:
        return 'cat-education'
    elif 'family' in cat_lower or 'child' in cat_lower:
        return 'cat-family'
    elif 'health' in cat_lower:
        return 'cat-health'
    return 'cat-sports'


def render_event_card(event, score=None):
    """Render a single event card."""
    category = event.get('category', 'Event')
    if pd.isna(category):
        category = 'Event'
    category_short = str(category).split(',')[0]
    cat_class = get_category_class(category)
    
    is_free = str(event.get('fee', '')).lower() != 'true'
    
    # Build the card HTML
    score_html = ""
    if score is not None and score > 0:
        score_pct = min(int(score * 100), 99)
        score_html = f'<span class="match-score">{score_pct}% match</span>'
    
    html = f"""
    <div class="event-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;">
            <span class="category-tag {cat_class}">{category_short}</span>
            <div>
                <span class="category-tag {'free-tag' if is_free else 'paid-tag'}">
                    {'Free' if is_free else 'Paid'}
                </span>
                {score_html}
            </div>
        </div>
        <div class="event-title">{event.get('event_name', 'Untitled Event')}</div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.75rem; line-height: 1.5;">
            {str(event.get('event_description', ''))[:200]}...
        </p>
        <div class="event-meta">
            📍 {event.get('analysis_neighborhood', 'San Francisco')} · 
            🕐 {format_time(event.get('start_time'))} · 
            👥 {str(event.get('age_group_eligibility_tags', 'All Ages')).split(';')[0]}
        </div>
    </div>
    """
    return html


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎉 SF Events Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-Powered Event Discovery · 2,075 San Francisco Events</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and model
    try:
        df = load_data()
        model, training_stats = get_trained_model(df)
    except FileNotFoundError:
        st.error("📁 Data file not found. Make sure `data/events.csv` exists.")
        st.info("Run the setup script or place the CSV file in the data folder.")
        return
    
    # Sidebar - Model Info
    with st.sidebar:
        st.markdown("### 🤖 Model Info")
        st.markdown(f"""
        **Algorithm:** TF-IDF + Feature Extraction
        
        **Training Data:**
        - {training_stats['n_events']:,} events
        - {training_stats['vocabulary_size']:,} vocabulary terms
        - {training_stats['n_features']:,} TF-IDF features
        
        **Features Extracted:**
        - Age group (kids, teens, families, adults)
        - Time of day (morning, afternoon, evening)
        - Cost (free/paid)
        - Category (sports, arts, education, etc.)
        - Weekend preference
        """)
        
        st.markdown("---")
        st.markdown("### 📊 How It Works")
        st.markdown("""
        1. **TF-IDF Vectorization** - Converts text to numerical vectors
        2. **Cosine Similarity** - Measures query-event similarity
        3. **Feature Boosting** - Adds points for matching filters
        4. **Ranking** - Returns top matches by combined score
        """)
    
    # Search Section
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search for events",
            placeholder="e.g., outdoor activities for kids this weekend...",
            label_visibility="collapsed"
        )
    with col2:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Sample queries
    st.markdown("**Try:** ", unsafe_allow_html=True)
    sample_cols = st.columns(5)
    samples = [
        "Fun activities for kids",
        "Free outdoor events", 
        "Art classes for teens",
        "Morning activities",
        "Weekend family events"
    ]
    for i, sample in enumerate(samples):
        if sample_cols[i].button(sample, key=f"sample_{i}"):
            query = sample
            search_clicked = True
    
    st.markdown("---")
    
    # Filters
    with st.expander("🎛️ Filters (optional)"):
        filter_cols = st.columns(4)
        with filter_cols[0]:
            category_filter = st.selectbox(
                "Category",
                ["All"] + sorted(df['category'].dropna().unique().tolist())
            )
        with filter_cols[1]:
            neighborhood_filter = st.selectbox(
                "Neighborhood", 
                ["All"] + sorted(df['analysis_neighborhood'].dropna().unique().tolist())
            )
        with filter_cols[2]:
            age_filter = st.selectbox(
                "Age Group",
                ["All", "Kids", "Teens", "Families", "Adults"]
            )
        with filter_cols[3]:
            free_only = st.checkbox("Free events only")
    
    # Search Results
    if query or search_clicked:
        with st.spinner("🔍 Searching..."):
            results = model.search(query if query else "activities", top_k=20)
            
            # Apply additional filters
            if category_filter != "All":
                results = results[results['category'].str.contains(category_filter, na=False)]
            if neighborhood_filter != "All":
                results = results[results['analysis_neighborhood'] == neighborhood_filter]
            if free_only:
                results = results[results['fee'].astype(str).str.lower() != 'true']
            if age_filter != "All":
                age_map = {'Kids': 'child', 'Teens': 'teen', 'Families': 'famil', 'Adults': 'tay'}
                results = results[results['age_group_eligibility_tags'].str.lower().str.contains(age_map.get(age_filter, ''), na=False)]
        
        # Display results
        if len(results) > 0:
            st.markdown(f"### Found {len(results)} events")
            
            # Results grid
            cols = st.columns(2)
            for i, (_, event) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(
                        render_event_card(event.to_dict(), event.get('match_score', 0)),
                        unsafe_allow_html=True
                    )
                    
                    # Expandable details
                    with st.expander("View details"):
                        st.markdown(f"**{event.get('event_name')}**")
                        st.markdown(f"*{event.get('org_name', 'Unknown organizer')}*")
                        st.markdown(event.get('event_description', ''))
                        
                        detail_cols = st.columns(2)
                        with detail_cols[0]:
                            st.markdown(f"📍 **Location:** {event.get('site_location_name', 'TBD')}")
                            st.markdown(f"🏘️ **Neighborhood:** {event.get('analysis_neighborhood', 'SF')}")
                            st.markdown(f"📅 **Date:** {str(event.get('event_start_date', 'TBD')).split()[0]}")
                        with detail_cols[1]:
                            st.markdown(f"🕐 **Time:** {format_time(event.get('start_time'))} - {format_time(event.get('end_time'))}")
                            st.markdown(f"👥 **Ages:** {event.get('age_group_eligibility_tags', 'All Ages')}")
                            st.markdown(f"📞 **Phone:** {event.get('site_phone', 'N/A')}")
                        
                        # Action buttons
                        btn_cols = st.columns(2)
                        address = event.get('site_address', '')
                        if address:
                            maps_url = f"https://www.google.com/maps/search/?api=1&query={address} San Francisco"
                            btn_cols[0].link_button("🗺️ Get Directions", maps_url)
                        
                        info_url = event.get('more_info', '')
                        if info_url:
                            if not info_url.startswith('http'):
                                info_url = f"https://{info_url}"
                            btn_cols[1].link_button("🔗 More Info", info_url)
        else:
            st.warning("No events found matching your search. Try different keywords or adjust filters.")
    
    else:
        # Show some default events
        st.markdown("### 🌟 Featured Events")
        sample_events = df.sample(min(6, len(df)))
        cols = st.columns(2)
        for i, (_, event) in enumerate(sample_events.iterrows()):
            with cols[i % 2]:
                st.markdown(render_event_card(event.to_dict()), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
