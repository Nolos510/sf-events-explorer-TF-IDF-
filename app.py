"""
SF Events Explorer - ML-Powered Event Discovery
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page config
st.set_page_config(page_title="SF Events Explorer", page_icon="🎉", layout="wide")

# ============================================================================
# ML MODEL
# ============================================================================
@st.cache_resource
def load_and_train():
    """Load data and train TF-IDF model."""
    # Try multiple paths - including root level
    paths_to_try = [
        'events.csv',           # Root level (your current structure)
        'data/events.csv',      # Standard structure
        './events.csv',
        './data/events.csv'
    ]
    
    df = None
    for path in paths_to_try:
        try:
            df = pd.read_csv(path)
            st.sidebar.success(f"✓ Loaded data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        st.error("❌ Could not find events.csv")
        st.info("Make sure events.csv is in your repository")
        st.stop()
    
    # Create search text
    df['search_text'] = (
        df['event_name'].fillna('') + ' ' +
        df['event_description'].fillna('') + ' ' +
        df['category'].fillna('')
    )
    
    # Train TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])
    
    return df, vectorizer, tfidf_matrix


def extract_features(query):
    """Extract structured features from query."""
    q = query.lower()
    features = {}
    
    if re.search(r'\b(kid|kids|child|children|toddler)\b', q):
        features['age'] = 'kids'
    elif re.search(r'\b(teen|teens|youth)\b', q):
        features['age'] = 'teens'
    elif re.search(r'\b(family|families)\b', q):
        features['age'] = 'families'
    
    if re.search(r'\b(morning|am)\b', q):
        features['time'] = 'morning'
    elif re.search(r'\b(afternoon)\b', q):
        features['time'] = 'afternoon'
    elif re.search(r'\b(evening|night)\b', q):
        features['time'] = 'evening'
    
    if re.search(r'\b(free)\b', q):
        features['free'] = True
    
    if re.search(r'\b(weekend|saturday|sunday)\b', q):
        features['weekend'] = True
    
    return features


def search_events(query, df, vectorizer, tfidf_matrix, top_k=15):
    """Search for matching events using TF-IDF + feature boosting."""
    # TF-IDF similarity
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Feature extraction and boosting
    features = extract_features(query)
    boosts = np.zeros(len(df))
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Age boost
        if features.get('age'):
            ages = str(row.get('age_group_eligibility_tags', '')).lower()
            if features['age'] == 'kids' and ('child' in ages or 'pre-teen' in ages or 'toddler' in ages):
                boosts[i] += 0.15
            elif features['age'] == 'teens' and ('teen' in ages or 'tay' in ages):
                boosts[i] += 0.15
            elif features['age'] == 'families' and ('famil' in ages or 'all' in ages):
                boosts[i] += 0.15
        
        # Free boost
        if features.get('free') and str(row.get('fee', '')).lower() != 'true':
            boosts[i] += 0.15
        
        # Time boost
        if features.get('time'):
            try:
                hour = int(str(row.get('start_time', '12:00')).split(':')[0])
                if features['time'] == 'morning' and hour < 12:
                    boosts[i] += 0.1
                elif features['time'] == 'afternoon' and 12 <= hour < 17:
                    boosts[i] += 0.1
                elif features['time'] == 'evening' and hour >= 17:
                    boosts[i] += 0.1
            except:
                pass
        
        # Weekend boost
        if features.get('weekend'):
            days = str(row.get('days_of_week', '')).lower()
            if 'sa' in days or 'su' in days:
                boosts[i] += 0.1
    
    # Combine scores
    final_scores = scores + boosts
    top_idx = np.argsort(final_scores)[::-1][:top_k]
    
    results = df.iloc[top_idx].copy()
    results['score'] = final_scores[top_idx]
    return results[results['score'] > 0.01]


def format_time(t):
    """Format time to 12-hour."""
    if pd.isna(t) or not t:
        return "TBD"
    try:
        h, m = str(t).split(':')[:2]
        h = int(h)
        return f"{h % 12 or 12}:{m} {'PM' if h >= 12 else 'AM'}"
    except:
        return str(t)


# ============================================================================
# APP UI
# ============================================================================
st.title("🎉 SF Events Explorer")
st.caption("ML-Powered Event Discovery • San Francisco")

# Load model
with st.spinner("🔄 Training ML model..."):
    df, vectorizer, tfidf_matrix = load_and_train()

st.success(f"✅ Model ready! Trained on **{len(df):,} events** with **{len(vectorizer.vocabulary_):,} vocabulary terms**")

st.markdown("---")

# Search
query = st.text_input("🔍 What are you looking for?", placeholder="e.g., outdoor activities for kids")

# Sample queries
cols = st.columns(5)
samples = ["Fun for kids", "Free events", "Art classes", "Morning activities", "Weekend family"]
for i, s in enumerate(samples):
    if cols[i].button(s, key=f"s{i}"):
        query = s

# Filters
with st.expander("🎛️ Filters"):
    fcols = st.columns(3)
    cat_filter = fcols[0].selectbox("Category", ["All"] + sorted(df['category'].dropna().unique().tolist()))
    hood_filter = fcols[1].selectbox("Neighborhood", ["All"] + sorted(df['analysis_neighborhood'].dropna().unique().tolist()))
    free_filter = fcols[2].checkbox("Free only")

st.markdown("---")

# Results
if query:
    results = search_events(query, df, vectorizer, tfidf_matrix)
    
    # Apply filters
    if cat_filter != "All":
        results = results[results['category'].str.contains(cat_filter, na=False)]
    if hood_filter != "All":
        results = results[results['analysis_neighborhood'] == hood_filter]
    if free_filter:
        results = results[results['fee'].astype(str).str.lower() != 'true']
    
    st.subheader(f"Found {len(results)} events")
    
    for _, row in results.iterrows():
        is_free = str(row.get('fee', '')).lower() != 'true'
        score_pct = int(min(row['score'] * 100, 99))
        
        with st.container():
            st.markdown(f"### {row['event_name']}")
            st.caption(f"📍 {row.get('analysis_neighborhood', 'SF')} • 🕐 {format_time(row.get('start_time'))} • {'🆓 Free' if is_free else '💰 Paid'} • 🎯 {score_pct}% match")
        
            with st.expander("View details"):
                st.write(row.get('event_description', 'No description'))
                st.markdown(f"""
                - **Location**: {row.get('site_location_name', 'TBD')}
                - **Address**: {row.get('site_address', 'N/A')}
                - **Date**: {str(row.get('event_start_date', 'TBD')).split()[0]}
                - **Time**: {format_time(row.get('start_time'))} - {format_time(row.get('end_time'))}
                - **Ages**: {row.get('age_group_eligibility_tags', 'All Ages')}
                """)
                
                bcols = st.columns(2)
                addr = row.get('site_address', '')
                if addr and pd.notna(addr):
                    bcols[0].link_button("🗺️ Directions", f"https://www.google.com/maps/search/?api=1&query={addr} San Francisco")
                info = row.get('more_info', '')
                if info and pd.notna(info):
                    url = info if str(info).startswith('http') else f"https://{info}"
                    bcols[1].link_button("🔗 More Info", url)
            
            st.divider()

else:
    st.info("👆 Enter a search query to find events!")
    st.subheader("🌟 Sample Events")
    for _, row in df.sample(min(5, len(df))).iterrows():
        is_free = str(row.get('fee', '')).lower() != 'true'
        st.markdown(f"**{row['event_name']}** — {row.get('analysis_neighborhood', 'SF')} {'🆓' if is_free else ''}")

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 ML Model Info")
    st.markdown(f"""
    **Algorithm**: TF-IDF + Feature Boosting
    
    **Training Stats**:
    - Events: {len(df):,}
    - Vocabulary: {len(vectorizer.vocabulary_):,} terms
    - Features: {tfidf_matrix.shape[1]:,}
    
    **Query Features Detected**:
    - Age group (kids, teens, families)
    - Time of day
    - Free/paid
    - Weekend
    """)
    
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. **TF-IDF** vectorizes query
    2. **Cosine similarity** scores events  
    3. **Feature extraction** boosts matches
    4. **Ranking** returns top results
    """)
    
    st.markdown("---")
    st.caption("SFSU Data Science Project")
