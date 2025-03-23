from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import sqlite3
import requests
from pytrends.request import TrendReq
from google.cloud import language_v1
import google.generativeai as genai
import logging
import pandas as pd
import matplotlib.pyplot as plt
from flask_socketio import SocketIO
import random

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = ""

DATABASE = 'imdb.db'

def create_database():
    """Initialize the database if not already created."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Movie Reviews Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        movie_title TEXT NOT NULL,
        rating INTEGER NOT NULL,
        review TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        latitude REAL,
        longitude REAL,
        district TEXT,
        state TEXT NOT NULL,
        nation TEXT,
        sentiment_score REAL,
        spam_label TEXT
    )
    """)

    # InfluenceIQ Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS influence_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        credibility_score REAL DEFAULT 0,
        fame_longevity REAL DEFAULT 0,
        engagement_quality REAL DEFAULT 0,
        influence_score REAL DEFAULT 0,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

# Ensure database is created once
create_database()

@app.route("/")
def index():
    """Render the index page with reviews, spam classifications, and InfluenceIQ rankings."""
    conn_imdb = sqlite3.connect(DATABASE)
    cursor_imdb = conn_imdb.cursor()

    # Fetch Reviews from `imdb.db`
    cursor_imdb.execute("""
        SELECT movie_title, rating, review, timestamp, district, state, nation, latitude, longitude, sentiment_score, spam_label 
        FROM reviews
    """)
    reviews = cursor_imdb.fetchall()

    # Fetch InfluenceIQ Rankings from `imdb.db`
    cursor_imdb.execute("""
        SELECT name, credibility_score, fame_longevity, engagement_quality, influence_score 
        FROM influence_scores ORDER BY influence_score DESC
    """)
    rankings = cursor_imdb.fetchall()

    conn_imdb.close()

    # Fetch Classified Reviews from `classify.db`
    conn_classify = sqlite3.connect("classify.db")
    cursor_classify = conn_classify.cursor()

    cursor_classify.execute("""
        SELECT review_id, review, spam_label FROM classified_reviews
    """)
    classified_reviews = cursor_classify.fetchall()

    conn_classify.close()

    return render_template("index.html", reviews=reviews, rankings=rankings, classified_reviews=classified_reviews)

@app.route('/submit_review', methods=['POST'])
def submit_review():
    """Submit a new movie review."""
    data = request.get_json()

    movie_title = data.get('movieTitle')
    rating = data.get('rating')
    review = data.get('review')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    district = data.get('district')
    state = data.get('state')
    nation = data.get('nation')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reviews (movie_title, rating, review, timestamp, latitude, longitude, district, state, nation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (movie_title, rating, review, timestamp, latitude, longitude, district, state, nation))

    conn.commit()
    conn.close()

    return jsonify({"message": "Review submitted successfully!"})





CX_ID = ""
SHEET_ID = ""


### 🔹 Fetch Online Mentions (Google Custom Search)
def get_online_mentions(public_figure_name):
    url = f"https://www.googleapis.com/customsearch/v1?q={public_figure_name}&cx={CX_ID}&key={API_KEY}"
    response = requests.get(url).json()
    
    # Extract total search results count
    if 'searchInformation' in response and 'totalResults' in response['searchInformation']:
        return int(response['searchInformation']['totalResults'])  # Convert to integer
    else:
        return 0  # No mentions found


def verify_identity(public_figure_name):
    """Verify if the public figure has a confirmed identity using Wikipedia and Google Search API."""
    
    # ✅ Step 1: Check Wikipedia API
    wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={public_figure_name.replace(' ', '_')}&format=json"
    wiki_response = requests.get(wiki_url).json()
    
    if "missing" not in str(wiki_response):  # If a Wikipedia page exists
        return True  # Verified ✅

    # ✅ Step 2: Check Google Search API for trusted mentions
    search_url = f"https://www.googleapis.com/customsearch/v1?q={public_figure_name}&key={GOOGLE_API_KEY}&cx={CX_ID}"
    search_response = requests.get(search_url).json()

    if "items" in search_response and len(search_response["items"]) > 0:
        # Check if any result comes from trusted sites (Wikipedia, IMDb, Forbes, ESPN)
        trusted_sites = ["wikipedia.org", "imdb.com", "forbes.com", "espn.com"]
        for item in search_response["items"]:
            if any(site in item["link"] for site in trusted_sites):
                return True  # Verified ✅

    return False  # Not Verified ❌



def get_youtube_engagement(public_figure_name):
    """Fetch all YouTube engagement metrics dynamically (no video count restriction)."""
    YOUTUBE_API_KEY = ""  # Replace with your actual API key
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    YOUTUBE_VIDEO_STATS_URL = "https://www.googleapis.com/youtube/v3/videos"

    video_ids = []
    next_page_token = None

    while True:
        search_params = {
            "part": "snippet",
            "q": public_figure_name,
            "maxResults": 50,  # Max limit per request
            "type": "video",
            "key": YOUTUBE_API_KEY,
            "pageToken": next_page_token  # Fetch next set of results
        }

        search_response = requests.get(YOUTUBE_SEARCH_URL, params=search_params).json()
        print("Search Response:", search_response)  # Debugging line

        # Extract video IDs
        for item in search_response.get("items", []):
            if "id" in item and "videoId" in item["id"]:
                video_ids.append(item["id"]["videoId"])

        # Check if there's another page of results
        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:  # No more pages
            break

    if not video_ids:
        return {"average_views": 0, "average_likes": 0, "average_comments": 0, "total_videos": 0}

    print("Video IDs:", video_ids)  # Debugging line

    # **Fetch statistics for all videos (process in chunks of 50 due to API limits)**
    total_views = total_likes = total_comments = 0
    total_videos = len(video_ids)

    for i in range(0, len(video_ids), 50):  # Process in batches of 50
        stats_params = {
            "part": "statistics",
            "id": ",".join(video_ids[i:i+50]),  # Get 50 video IDs at a time
            "key": YOUTUBE_API_KEY
        }

        stats_response = requests.get(YOUTUBE_VIDEO_STATS_URL, params=stats_params).json()
        print("Stats Response:", stats_response)  # Debugging line

        # Sum up views, likes, and comments
        for video in stats_response.get("items", []):
            stats = video.get("statistics", {})
            total_views += int(stats.get("viewCount", 0))
            total_likes += int(stats.get("likeCount", 0))
            total_comments += int(stats.get("commentCount", 0))

    num_videos = max(1, total_videos)  # Prevent division by zero

    return {
        "average_views": total_views // num_videos,
        "average_likes": total_likes // num_videos,
        "average_comments": total_comments // num_videos,
        "total_videos": total_videos  # Now dynamically fetched
    }

### 🔹 Log Influence Data to Google Sheets
def log_to_google_sheets(name, credibility, longevity, engagement, influence_score, mentions, youtube, verified):
    sheet_url = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/A1:append?valueInputOption=RAW"
    data = {
        "values": [[name, credibility, longevity, engagement, influence_score, mentions, youtube, "Yes" if verified else "No", str(datetime.now())]]
    }
    requests.post(sheet_url, json=data, headers={"Authorization": f"Bearer {GOOGLE_API_KEY}"})


### 🔹 Fully Automated InfluenceIQ Submission
@app.route('/submit_influence_review', methods=['POST'])
def submit_influence_review():
    """Submit InfluenceIQ review based on real data (No manual scores)."""
    data = request.get_json()
    name = data.get('name')

    # **Fetch Influence Data**
    online_mentions = get_online_mentions(name)
    youtube_engagement = get_youtube_engagement(name)  # This is a dict
    is_verified = verify_identity(name)

    # **Extract engagement data**
    avg_views = youtube_engagement.get('average_views', 0)  # Default to 0 if missing
    avg_likes = youtube_engagement.get('average_likes', 0)
    avg_comments = youtube_engagement.get('average_comments', 0)

    # **Automated Score Calculation**
    credibility_score = 50 + (20 if is_verified else 0) + (online_mentions * 0.2)  # Base: 50, Verified Bonus: 20, Mentions Boost
    fame_longevity = 50 + (online_mentions * 0.5)  # Base: 50, Mentions Weighted

    # **Fixed engagement calculation**
    engagement_quality = 50 + ((avg_views * 0.00001) + (avg_likes * 0.001) + (avg_comments * 0.1))  # Normalized

    credibility_score = min(100, credibility_score)
    fame_longevity = min(100, fame_longevity)
    engagement_quality = min(100, engagement_quality)

    # **Final Influence Score Calculation (Weighted)**
    influence_score = (credibility_score * 0.4) + (fame_longevity * 0.3) + (engagement_quality * 0.3)

    # **Store in SQLite**
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM influence_scores WHERE name=?", (name,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE influence_scores 
            SET credibility_score=?, fame_longevity=?, engagement_quality=?, influence_score=?, 
                online_mentions=?, youtube_engagement=?, verified=?, last_updated=CURRENT_TIMESTAMP 
            WHERE name=?
        """, (credibility_score, fame_longevity, engagement_quality, influence_score, 
            online_mentions, str(youtube_engagement), "Yes ✅" if is_verified else "No ❌", name))
    else:
        cursor.execute("""
            INSERT INTO influence_scores (name, credibility_score, fame_longevity, engagement_quality, 
                                        influence_score, online_mentions, youtube_engagement, verified) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, credibility_score, fame_longevity, engagement_quality, 
            influence_score, online_mentions, str(youtube_engagement), "Yes ✅" if is_verified else "No ❌"))

    conn.commit()
    conn.close()

    # **Log to Google Sheets**
    log_to_google_sheets(name, credibility_score, fame_longevity, engagement_quality, influence_score, online_mentions, youtube_engagement, is_verified)

    return jsonify({
    "message": "InfluenceIQ review submitted!",
    "influence_score": influence_score,
    "online_mentions": online_mentions,
    "youtube_engagement": {
        "average_views": avg_views,
        "average_likes": avg_likes,
        "average_comments": avg_comments,
        "total_videos": youtube_engagement.get('total_videos', 0)
    },
    "is_verified": is_verified
})




@app.route('/get_influence_ranking', methods=['GET'])
def get_influence_ranking():
    """Fetch the ranked list of public figures based on InfluenceIQ score."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT name, credibility_score, fame_longevity, engagement_quality, influence_score 
    FROM influence_scores ORDER BY influence_score DESC
    """)
    rankings = cursor.fetchall()
    
    conn.close()

    return jsonify({"rankings": rankings})


# Replace with your actual API key
API_KEY = ""
# Configure the API key directly
genai.configure(api_key=API_KEY)
# Google Cloud API URLs
GOOGLE_NLP_URL = "https://language.googleapis.com/v1/documents:analyzeSentiment"
PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"


def analyze_text_nlp(text):
    """Analyzes review text using Gemini Flash 2.0 to detect spam."""
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Analyze the following review and classify it as either 'Spam' or 'Not Spam'.
    
    Review: "{text}"
    
    Provide a concise answer: 'Spam' or 'Not Spam'.
    """
    
    response = model.generate_content(prompt)
    spam_label = response.text.strip()

    return spam_label


def create_classify_db():
    """Creates classify.db only if it does not exist."""
    conn = sqlite3.connect("classify.db")
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classified_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER UNIQUE,
            review TEXT,
            spam_label TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def get_last_review_id():
    """Fetches the last stored review ID to avoid duplicate storage."""
    with sqlite3.connect("classify.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(review_id) FROM classified_reviews")
        last_id = cursor.fetchone()[0]
    
    return last_id if last_id is not None else 0

def detect_spam_reviews():
    """Detects spam/manipulative reviews and stores results in classify.db."""
    create_classify_db()  # Ensure DB is created
    
    conn = sqlite3.connect("imdb.db")
    cursor = conn.cursor()
    
    # Get last stored review_id from classify.db
    last_stored_id = get_last_review_id()

    # Fetch only new reviews from imdb.db
    cursor.execute("SELECT id, review FROM reviews WHERE id > ?", (last_stored_id,))
    new_reviews = cursor.fetchall()

    conn.close()

    if not new_reviews:
        print("No new reviews to classify.")
        return

    conn_classify = sqlite3.connect("classify.db")
    cursor_classify = conn_classify.cursor()

    for review_id, text in new_reviews:
        spam_label = analyze_text_nlp(text)  # Call your NLP function

        print(f"Review ID: {review_id}")
        print(f"Review: {text}")
        print(f"Spam Classification: {spam_label}")
        print("-" * 50)

        # Store classification results in classify.db
        cursor_classify.execute("""
            INSERT INTO classified_reviews (review_id, review, spam_label) 
            VALUES (?, ?, ?)
        """, (review_id, text, spam_label))

    conn_classify.commit()
    conn_classify.close()




GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

def analyze_review(review_text):
    """Analyzes review & detects spam using Gemini Flash 2.0"""


    ### 2️⃣ Spam Detection (Gemini Flash 2.0) ###
    gemini_prompt = f"""
    Analyze the following movie review and classify it as Spam or Not Spam. Spam includes fake promotions, excessive emojis, repeated words, clickbait, or offensive language.

    Review: "{review_text}"

    Respond with only 'Spam' or 'Not Spam'.
    """

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        gemini_response = gemini_model.generate_content(gemini_prompt)
        spam_result = gemini_response.text.strip().lower()

        is_spam = 1 if "spam" in spam_result else 0
    except Exception as e:
        print(f"Error in Gemini API: {e}")
        is_spam = 0  # Default to not spam if API fails

    return 0, is_spam

detect_spam_reviews()

def analyze_all_reviews():
    """Fetch all reviews from the database and classify them based on sentiment and spam score."""
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Fetch all reviews
    cursor.execute("SELECT id, review FROM reviews")
    reviews = cursor.fetchall()
    
    conn.close()

    classified_reviews = []

    for review_id, review_text in reviews:
        sentiment_score, spam_score = analyze_review(review_text)

        # Classify sentiment
        if sentiment_score > 0.3:
            sentiment_label = "Positive"
        elif sentiment_score < -0.3:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Classify spam
        spam_label = "Spam" if spam_score > 0.7 else "Not Spam"

        # Print results
        print(f"Review ID: {review_id}")
        print(f"Review: {review_text}")
        print(f"Sentiment: {sentiment_label} (Score: {sentiment_score})")
        print(f"Spam Classification: {spam_label} (Score: {spam_score})")
        print("-" * 50)

        classified_reviews.append((review_id, review_text, sentiment_label, spam_label))

    return classified_reviews

# Run the analysis on all stored reviews
#analyze_all_reviews()



def calculate_fame_longevity(name):
    """Calculate fame longevity using Google Trends."""
    pytrends = TrendReq()
    pytrends.build_payload([name], timeframe="today 5-y")
    trend_data = pytrends.interest_over_time()

    return trend_data[name].mean() / 100 if not trend_data.empty else 0  # Fixed return value

import pandas as pd

def ensure_fair_ai():
    """Checks for bias in credibility, fame longevity, and engagement quality scores."""
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT credibility_score, fame_longevity, engagement_quality FROM influence_scores", conn)
    conn.close()

    mean_values = df.mean()
    std_dev = df.std()
    bias_detected = any(std_dev / mean_values > 0.3)  # Arbitrary threshold for fairness check

    print("AI Bias Analysis:")
    print(f"Mean Values: \n{mean_values}")
    print(f"Standard Deviation: \n{std_dev}")
    print("Bias Detected!" if bias_detected else "No significant bias detected.")
    print("-" * 50)


import matplotlib.pyplot as plt

def generate_dashboard_insights():
    """Generates charts for InfluenceIQ analysis."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Fetch review data
    cursor.execute("SELECT rating FROM reviews")
    ratings = [row[0] for row in cursor.fetchall()]

    # Fetch influence data
    cursor.execute("SELECT credibility_score, fame_longevity, engagement_quality FROM influence_scores")
    influence_data = cursor.fetchall()

    conn.close()

    # Plot Ratings Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(ratings, bins=5, edgecolor="black", alpha=0.7)
    plt.xlabel("Ratings")
    plt.ylabel("Number of Reviews")
    plt.title("Movie Ratings Distribution")
    plt.show()

    # Plot Influence Scores
    labels = ["Credibility", "Fame Longevity", "Engagement"]
    values = [sum(x) / len(influence_data) for x in zip(*influence_data)] if influence_data else [0, 0, 0]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=["blue", "green", "red"])
    plt.xlabel("Metrics")
    plt.ylabel("Average Score")
    plt.title("InfluenceIQ Ranking Factors")
    plt.show()




def analyze_sentiment(text):
    """Analyzes sentiment using Google Cloud Natural Language API."""
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={"document": document}).document_sentiment
    return sentiment.score  # Returns a score between -1.0 (negative) to 1.0 (positive)

def update_influence_scores():
    """Updates credibility, fame longevity, and engagement dynamically."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM influence_scores")
    influencers = cursor.fetchall()

    for influencer_id, name in influencers:
        # Compute fame longevity
        fame_longevity = calculate_fame_longevity(name)

        # Compute engagement quality
        cursor.execute("SELECT review FROM reviews WHERE movie_title = ?", (name,))
        reviews = cursor.fetchall()
        engagement_score = sum(analyze_sentiment(review[0]) for review in reviews) / len(reviews) if reviews else 0

        # Compute credibility (example metric: engagement * longevity)
        credibility_score = (fame_longevity + engagement_score) / 2

        # Final InfluenceIQ Score
        influence_score = (credibility_score * 0.4) + (fame_longevity * 0.3) + (engagement_score * 0.3)

        # Update DB
        cursor.execute("""
            UPDATE influence_scores SET credibility_score=?, fame_longevity=?, engagement_quality=?, influence_score=?
            WHERE id=?
        """, (credibility_score, fame_longevity, engagement_score, influence_score, influencer_id))

    conn.commit()
    conn.close()

import time
from flask_socketio import SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")

# Simulating real-time spam vs. proper review counts
def generate_live_data():
    spam_count = 0
    proper_count = 0

    while True:
        time.sleep(2)  # Update every 2 seconds
        spam_count += random.randint(0, 5)
        proper_count += random.randint(0, 10)

        socketio.emit('update_chart', {
            'time': time.strftime('%H:%M:%S'),
            'spam': spam_count,
            'proper': proper_count
        })

# Run in background
import threading
threading.Thread(target=generate_live_data, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8500))  # Render assigns a PORT dynamically
    app.run(host="0.0.0.0", port=port)