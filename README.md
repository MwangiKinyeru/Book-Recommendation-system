# Book-Recommendation-system

## Overview
This Book Recommendation System provides personalized book recommendations based on user input. The system utilizes TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to identify books similar to a given title, author, or publisher.

## Features
>> •	Accepts user input as a book title, author, or publisher.
>> •	Finds the closest match in the dataset and retrieves similar books.
>> •	Uses TF-IDF vectorization for text processing.
>> •	Computes similarity using cosine similarity.
>> •	Provides clear feedback when a book is not found.

## How It Works

### 1.	Preprocessing
>> o	Converts book titles, authors, and publishers to lowercase.
>> o	Removes extra spaces and unwanted characters.
>> o	Combines these fields into a single feature for better search accuracy.

### 2.	Vectorization & Similarity Computation
>> o	Uses TF-IDF to convert text data into numerical vectors.
>> o	Computes cosine similarity to measure the similarity between books.

### 3.	Recommendation Process
>> o	Matches the user input with book details.
>> o	Retrieves the most similar books based on their content.
>> o	Displays up to 5 recommended books.

## Example Output
> Enter a book title, author, or publisher: da capo press
> Recommended books based on your input:
>> - Complete Stories Truman Capote
>> - In Cold Blood
>> - Summer Crossing
>> - Breakfast at Tiffany's

## Future Improvements
>> •	Implement Content-Based Filtering with Word Embeddings.
>> •	Add a User-Based Collaborative Filtering option.
>> •	Improve Search Functionality with fuzzy matching.
>> •	Deploy as a Web API or Flask/Django web app.
## Author
### Data Scientist Martin Waweru