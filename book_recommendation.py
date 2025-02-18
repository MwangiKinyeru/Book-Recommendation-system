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
book = pd.read_csv("Data\cleaned_books.csv", sep=",", on_bad_lines='skip', nrows=5000)
# Text Preprocessing
stop_words = set(stopwords.words('english'))



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