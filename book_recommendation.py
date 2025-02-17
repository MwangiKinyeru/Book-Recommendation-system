# Necessary Imports
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
book = pd.read_csv("Data/books.csv", sep=",", on_bad_lines='skip')

# Rename and drop irrelevant columns
book.rename(columns={"language_code": "language"}, inplace=True)
irrelevant_columns = ['bookID', 'average_rating', 'isbn', 'isbn13', '  num_pages', 'ratings_count', 'text_reviews_count', 'publication_date']
book.drop(columns=irrelevant_columns, inplace=True)

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def clean_title(title):
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    words = title.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def standardize_authors(authors):
    authors = authors.split("/")[0].strip()
    authors = re.sub(r'\s+', ' ', authors)
    return authors.lower()

def standardize_publisher(publisher):
    publisher = publisher.strip().lower()
    publisher = re.sub(r"\s*(inc\.|corporation|co\.|ltd\.|company|corp\.|\(.*\))\s*", "", publisher)
    publisher = re.sub(r'\s+', ' ', publisher)
    return publisher

def clean_language(language):
    language = language.strip().lower()
    language_mapping = {
        'en-us': 'en', 'en-gb': 'en', 'en-ca': 'en', 'en': 'en', 'eng': 'en',
        'fre': 'fr', 'fra': 'fr',
        'spa': 'es', 'esp': 'es',
        'ger': 'de', 'deu': 'de',
        'por': 'pt',
        'zho': 'zh',
        'jpn': 'ja',
        'rus': 'ru',
        'ita': 'it',
        'grc': 'el',
        'gla': 'ga',
        'mul': 'mix',
    }
    return language_mapping.get(language, 'unknown')

# Apply text preprocessing
book["title"] = book["title"].apply(clean_title)
book["authors"] = book["authors"].apply(standardize_authors)
book["publisher"] = book["publisher"].apply(standardize_publisher)
book["language"] = book["language"].apply(clean_language)

# Combine features for TF-IDF
book["combined_features"] = book["title"] + " " + book["authors"] + " " + book["publisher"] + " " + book["language"]

# Drop duplicates and reset index
book.drop_duplicates(inplace=True)
book = book.reset_index(drop=True)  # Reset the index

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(book["combined_features"])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations_based_on_input(user_input, cosine_sim=cosine_sim):
    user_input = user_input.lower()
    idx = book[book['title'].str.contains(user_input, case=False, na=False)].index
    if not idx.empty:
        input_type = 'title'
    else:
        idx = book[book['authors'].str.contains(user_input, case=False, na=False)].index
        if not idx.empty:
            input_type = 'authors'
        else:
            idx = book[book['publisher'].str.contains(user_input, case=False, na=False)].index
            input_type = 'publisher' if not idx.empty else None

    if idx.empty:
        return "❌ Book not found. Please try a different title, author, or publisher."
    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return book.iloc[book_indices]

# Streamlit App
st.title("📚 Book Recommendation System")
st.write("Enter a book title, author, or publisher to get recommendations.")

# Add a sidebar
# Add a sidebar
st.sidebar.title("About")
st.sidebar.write("""
### Welcome to the Book Recommendation System! 📚

This app helps you discover new books based on your preferences. Whether you're looking for a specific title, an author, or books from a particular publisher, this app has you covered.

#### How It Works:
1. **Search by**: Use the dropdown to select whether you want to search by **Title**, **Author**, or **Publisher**.
2. **Enter your input**: Type in the name of a book, author, or publisher.
3. **Get recommendations**: The app will suggest 10 books similar to your input.

#### Features:
- **Personalized Recommendations**: Get tailored book suggestions based on your input.
- **Easy to Use**: Simple and intuitive interface.
- **Multiple Search Options**: Search by title, author, or publisher.
""")
st.sidebar.write("Created by: DS. Martin Waweru")

# Add a dropdown for input type
input_type = st.selectbox("Search by:", ["Title", "Author", "Publisher"])
user_input = st.text_input(f"Enter a book {input_type.lower()}:")

if user_input:
    recommended_books = get_recommendations_based_on_input(user_input)
    if isinstance(recommended_books, str):
        st.error(recommended_books)  # Display error message if book not found
    else:
        st.write("### Recommended Books:")
        # Display book details in a clean format
        for i, book_details in recommended_books.iterrows():
            st.write(f"**Title:** {book_details['title']}")
            st.write(f"**Author:** {book_details['authors']}")
            st.write(f"**Publisher:** {book_details['publisher']}")
            st.write("---")