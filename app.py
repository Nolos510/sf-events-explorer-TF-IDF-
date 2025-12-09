"""
SF Events Explorer - ML-Powered Event Discovery
Beautiful UI Version
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page config
st.set_page_config(
    page_title="SF Events Explorer",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Event card styling */
    .event-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .event-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
    }
    
    .event-title {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .event-meta {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    
    .event-description {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Tags */
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .tag-category {
        background: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    .tag-free {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .tag-paid {
        background: rgba(100, 116, 139, 0.2);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
    
    .tag-match {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.3));
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 2px solid #475569 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Success message */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #34d399 !important;
    }
    
    /* Divider */
    hr {
        border-color: #334155 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 8px !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        border-color: #475569 !important;
    }
    
    /* Info box */
    .stInfo {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #a5b4fc !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ML MODEL
# ============================================================================
@st.cache_resource
def load_and_train():
    """Load data and train TF-IDF model."""
    paths_to_try = ['events.csv', 'data/events.csv', './events.csv', './data/events.csv']
    
    df = None
    for path in paths_to_try:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        st.error("❌ Could not find events.csv")
        st.stop()
    
    df['search_text'] = (
        df['event_name'].fillna('') + ' ' +
        df['event_description'].fillna('') + ' ' +
        df['category'].fillna('')
    )
    
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
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    features = extract_features(query)
    boosts = np.zeros(len(df))
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if features.get('age'):
            ages = str(row.get('age_group_eligibility_tags', '')).lower()
            if features['age'] == 'kids' and ('child' in ages or 'pre-teen' in ages or 'toddler' in ages):
                boosts[i] += 0.15
            elif features['age'] == 'teens' and ('teen' in ages or 'tay' in ages):
                boosts[i] += 0.15
            elif features['age'] == 'families' and ('famil' in ages or 'all' in ages):
                boosts[i] += 0.15
        
        if features.get('free') and str(row.get('fee', '')).lower() != 'true':
            boosts[i] += 0.15
        
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
        
        if features.get('weekend'):
            days = str(row.get('days_of_week', '')).lower()
            if 'sa' in days or 'su' in days:
                boosts[i] += 0.1
    
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


def get_category_emoji(category):
    """Get emoji for category."""
    if pd.isna(category):
        return "📌"
    cat = str(category).lower()
    if 'sport' in cat:
        return "⚽"
    elif 'art' in cat or 'culture' in cat:
        return "🎨"
    elif 'education' in cat:
        return "📚"
    elif 'family' in cat:
        return "👨‍👩‍👧‍👦"
    elif 'health' in cat:
        return "💪"
    return "📌"


def render_event_card(row, show_details=False):
    """Render an event card with custom HTML."""
    is_free = str(row.get('fee', '')).lower() != 'true'
    score_pct = int(min(row.get('score', 0) * 100, 99))
    category = str(row.get('category', 'Event')).split(',')[0] if pd.notna(row.get('category')) else 'Event'
    emoji = get_category_emoji(row.get('category'))
    
    # Build tags HTML
    tags_html = f"""
        <span class="tag tag-category">{emoji} {category}</span>
        <span class="tag {'tag-free' if is_free else 'tag-paid'}">{'🆓 Free' if is_free else '💰 Paid'}</span>
        <span class="tag tag-match">🎯 {score_pct}% match</span>
    """
    
    # Event meta info
    neighborhood = row.get('analysis_neighborhood', 'San Francisco')
    time_str = format_time(row.get('start_time'))
    age_group = str(row.get('age_group_eligibility_tags', 'All Ages')).split(';')[0].strip()
    
    card_html = f"""
    <div class="event-card">
        <div style="margin-bottom: 0.75rem;">
            {tags_html}
        </div>
        <div class="event-title">{row.get('event_name', 'Untitled Event')}</div>
        <div class="event-meta">
            📍 {neighborhood} &nbsp;•&nbsp; 🕐 {time_str} &nbsp;•&nbsp; 👥 {age_group}
        </div>
        <div class="event-description">
            {str(row.get('event_description', ''))[:200]}...
        </div>
    </div>
    """
    return card_html


# ============================================================================
# APP UI
# ============================================================================

# Load model
df, vectorizer, tfidf_matrix = load_and_train()

# Main content
st.markdown('<h1 class="main-title">🎉 SF Events Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML-Powered Event Discovery • 2,075 San Francisco Events</p>', unsafe_allow_html=True)

# Success message
st.success(f"✅ Model trained on **{len(df):,} events** with **{len(vectorizer.vocabulary_):,}** vocabulary terms")

st.markdown("---")

# Search section
st.markdown("### 🔍 What are you looking for?")
query = st.text_input("Search", placeholder="e.g., outdoor activities for kids, free concerts, art classes...", label_visibility="collapsed")

# Sample query buttons
st.markdown("**Quick searches:**")
cols = st.columns(5)
samples = ["🧒 Fun for kids", "🆓 Free events", "🎨 Art classes", "🌅 Morning activities", "👨‍👩‍👧 Weekend family"]
sample_queries = ["Fun for kids", "Free events", "Art classes", "Morning activities", "Weekend family"]

for i, (label, sq) in enumerate(zip(samples, sample_queries)):
    if cols[i].button(label, key=f"sample_{i}", use_container_width=True):
        query = sq

# Filters section
st.markdown("---")
with st.expander("🎛️ **Advanced Filters**", expanded=False):
    fcols = st.columns(3)
    with fcols[0]:
        categories = ["All Categories"] + sorted(df['category'].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", categories)
    with fcols[1]:
        neighborhoods = ["All Neighborhoods"] + sorted(df['analysis_neighborhood'].dropna().unique().tolist())
        hood_filter = st.selectbox("Neighborhood", neighborhoods)
    with fcols[2]:
        free_filter = st.checkbox("🆓 Free events only")

st.markdown("---")

# Results section
if query:
    with st.spinner("🔍 Searching..."):
        results = search_events(query, df, vectorizer, tfidf_matrix)
    
    # Apply filters
    if cat_filter != "All Categories":
        results = results[results['category'].str.contains(cat_filter, na=False)]
    if hood_filter != "All Neighborhoods":
        results = results[results['analysis_neighborhood'] == hood_filter]
    if free_filter:
        results = results[results['fee'].astype(str).str.lower() != 'true']
    
    st.markdown(f"### Found **{len(results)}** events")
    
    if len(results) == 0:
        st.warning("No events found. Try different keywords or adjust filters.")
    else:
        # Display results in a grid
        for idx, (_, row) in enumerate(results.iterrows()):
            st.markdown(render_event_card(row), unsafe_allow_html=True)
            
            # Details in columns (cleaner than expander)
            with st.expander(f"📋 View details for: {row['event_name'][:50]}..."):
                dcols = st.columns(2)
                with dcols[0]:
                    st.markdown(f"**🏢 Organization:** {row.get('org_name', 'N/A')}")
                    st.markdown(f"**📍 Location:** {row.get('site_location_name', 'TBD')}")
                    st.markdown(f"**🗺️ Address:** {row.get('site_address', 'N/A')}")
                    st.markdown(f"**🏘️ Neighborhood:** {row.get('analysis_neighborhood', 'SF')}")
                with dcols[1]:
                    st.markdown(f"**📅 Date:** {str(row.get('event_start_date', 'TBD')).split()[0]}")
                    st.markdown(f"**🕐 Time:** {format_time(row.get('start_time'))} - {format_time(row.get('end_time'))}")
                    st.markdown(f"**👥 Ages:** {row.get('age_group_eligibility_tags', 'All Ages')}")
                    st.markdown(f"**📞 Phone:** {row.get('site_phone', 'N/A')}")
                
                st.markdown("**📝 Full Description:**")
                st.write(row.get('event_description', 'No description available.'))
                
                # Action buttons
                bcols = st.columns(2)
                addr = row.get('site_address', '')
                if addr and pd.notna(addr):
                    bcols[0].link_button("🗺️ Get Directions", f"https://www.google.com/maps/search/?api=1&query={addr} San Francisco")
                info = row.get('more_info', '')
                if info and pd.notna(info):
                    url = info if str(info).startswith('http') else f"https://{info}"
                    bcols[1].link_button("🔗 More Info", url)

else:
    # Landing state - show featured events
    st.markdown("### 🌟 Featured Events")
    st.markdown("*Enter a search query above to find events, or browse some featured picks:*")
    
    # Show random sample
    sample_events = df.sample(min(6, len(df)))
    for _, row in sample_events.iterrows():
        is_free = str(row.get('fee', '')).lower() != 'true'
        category = str(row.get('category', '')).split(',')[0] if pd.notna(row.get('category')) else 'Event'
        emoji = get_category_emoji(row.get('category'))
        
        st.markdown(f"""
        <div class="event-card">
            <span class="tag tag-category">{emoji} {category}</span>
            <span class="tag {'tag-free' if is_free else 'tag-paid'}">{'🆓 Free' if is_free else '💰 Paid'}</span>
            <div class="event-title">{row['event_name']}</div>
            <div class="event-meta">📍 {row.get('analysis_neighborhood', 'SF')}</div>
        </div>
        """, unsafe_allow_html=True)
