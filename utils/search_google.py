import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np

# Function to get search results from Google
def get_search_results(query):
    urls = [
        "https://en.wikipedia.org/wiki/Narendra_Modi",
        "https://www.pmindia.gov.in/en/",
        # Add more URLs as needed
    ]
    web_texts = [
        "Narendra Damodardas Modi is an Indian politician serving as the 14th and current prime minister of India since 2014.",
        "Official website of the Prime Minister of India.",
        # Add more web text as needed
    ]
    return urls, web_texts

# Function to extract passages from URLs
def extract_passages(urls):
    passages = []
    for url in urls:
        # Fetch web page content using requests
        response = requests.get(url)
        if response.status_code == 200:
            # Parse HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text passages from HTML content
            text = soup.get_text()
            # Split passages into sentences using nltk.sent_tokenize
            sentences = sent_tokenize(text)
            # Add sentences to passages list
            passages.extend(sentences)
    return passages

from sentence_transformers import SentenceTransformer, util

# Load pre-trained Sentence Transformers model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Function to calculate relevance score using Sentence Transformers
def calculate_relevance(query, passages):
    # Encode query and passages into embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)

    # Calculate cosine similarity between query and passages
    similarity_scores = util.pytorch_cos_sim(query_embedding, passage_embeddings)

    # Convert similarity scores to relevance scores
    relevance_scores = similarity_scores.tolist()

    return relevance_scores

import numpy as np

# Function to return most relevant answer based on BERT relevance scores
def get_most_relevant_answer(query, relevance_scores, passages):
    # Find index of passage with highest relevance score
    most_relevant_index = np.argmax(relevance_scores)
    # Return most relevant answer from passages
    return passages[most_relevant_index]


# Main function
def main(query):
    # Get search results from Google
    urls, _ = get_search_results(query)
    # Extract passages from URLs
    passages = extract_passages(urls)
    # Calculate relevance score
    relevance_scores = calculate_relevance(query, passages)
    # Get most relevant answer
    answer = get_most_relevant_answer(query, relevance_scores, passages)
    return answer

# Example usage
query = "Who is the PM of India?"
answer = main(query)
print("Most relevant answer:", answer)
