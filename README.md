# SF Events Explorer

**ML-Powered Event Discovery for San Francisco**

A Streamlit app that uses TF-IDF machine learning to match natural language queries to 2,075 SF events.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)

## Live Demo

**[Your Streamlit Cloud URL will be here]**

---

## The ML Model

### Algorithm: TF-IDF + Feature Extraction

This app trains a **TfidfVectorizer** on the event corpus to enable semantic search:

```python
# Training (happens on app startup)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    stop_words='english'
)
tfidf_matrix = vectorizer.fit_transform(event_texts)  # Learn vocabulary + IDF weights


### How Search Works

1. **TF-IDF Vectorization** - User query → numerical vector
2. **Cosine Similarity** - Compare query vector to all 2,075 event vectors
3. **Feature Extraction** - Parse query for age, time, cost, category preferences
4. **Score Boosting** - Add bonus points for matching structured features
5. **Ranking** - Return top matches sorted by combined score

### Feature Extraction

The model extracts these features from natural language:

| Feature | Example Query | Detected Value |
|---------|--------------|----------------|
| Age Group | "activities for **kids**" | `kids` |
| Time | "**morning** events" | `morning` |
| Cost | "**free** things to do" | `is_free=True` |
| Category | "**art** classes" | `Arts, Culture & Identity` |
| Weekend | "**Saturday** activities" | `is_weekend=True` |

### Model Statistics

- **Training Data**: 2,075 SF events
- **Vocabulary Size**: ~3,000 terms
- **Features**: 5,000 TF-IDF dimensions
- **Inference Time**: <100ms per query


## Project Structure

```
sf-events-app/
├── app.py                 # Main Streamlit app + ML model
├── requirements.txt       # Python dependencies
├── data/
│   └── events.csv        # 2,075 SF events dataset
├── .streamlit/
│   └── config.toml       # Theme configuration
└── README.md
```

---

## Local Development

### Prerequisites
- Python 3.9+
- pip

App will be live at:
```
(https://sfeventexplorer574.streamlit.app/)```

## Data Source

- **Dataset**: Our415 Events & Activities
- **Source**: San Francisco Open Data
- **Records**: 2,075 events
- **Updated**: October 7, 2025
- **License**: Public Domain (Government Data)

### Data Fields Used

| Field | Description |
|-------|-------------|
| `event_name` | Event title |
| `event_description` | Full description |
| `category` | Sports, Arts, Education, etc. |
| `age_group_eligibility_tags` | Kids, Teens, Families, etc. |
| `start_time` / `end_time` | Event timing |
| `fee` | true/false for paid events |
| `analysis_neighborhood` | SF neighborhood |
| `site_address` | Location address |

---

## Team

| Role | Name |
|------|------|
| Product | Nhi |
| Data | JD |
| Modeling | Charley |
| Prototype | Carlo |

---

## Future Enhancements

- [ ] Budget/price filtering (pending Yelp integration)
- [ ] Drag-and-drop agenda builder
- [ ] User accounts + saved events
- [ ] Calendar export (iCal)
- [ ] Mobile app version


---

SFSU Data Science Project • San Francisco Events Discovery
