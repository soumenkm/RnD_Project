"""Utils for searching a query and returning top passages from search results."""
import concurrent.futures
import itertools
import os, sys, time
sys.path.append("/root/RnD_Project/utils")
import random
from typing import Any, Dict, List, Tuple

import bs4
import requests
import spacy
import torch
from sentence_transformers import CrossEncoder
import api
# import cohere_call

PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device="cpu",
)
SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search/"
# SUBSCRIPTION_KEY = os.getenv("AZURE_SEARCH_KEY")
# print(SUBSCRIPTION_KEY)
TOKENIZER = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
# python -m spacy download en_core_web_sm

def chunk_text(
    text: str,
    sentences_per_passage: int,
    filter_sentence_len: int,
    sliding_distance: int = None,
) -> List[str]:
    """Chunks text into passages using a sliding window.

    Args:
        text: Text to chunk into passages.
        sentences_per_passage: Number of sentences for each passage.
        filter_sentence_len: Maximum number of chars of each sentence before being filtered.
        sliding_distance: Sliding distance over the text. Allows the passages to have
            overlap. The sliding distance cannot be greater than the window size.
    Returns:
        passages: Chunked passages from the text.
    """
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    assert sentences_per_passage > 0 and sliding_distance > 0

    passages = []
    try:
        doc = TOKENIZER(text[:500000])  # Take 500k chars to not break tokenization.
        sents = [
            s.text
            for s in doc.sents
            if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
        ]
        for idx in range(0, len(sents), sliding_distance):
            passages.append(" ".join(sents[idx : idx + sentences_per_passage]))
    except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
        print("Unicode error when using Spacy. Skipping text.")

    return passages


def is_tag_visible(element: bs4.element) -> bool:
    """Determines if an HTML element is visible.

    Args:
        element: A BeautifulSoup element to check the visiblity of.
    returns:
        Whether the element is visible.
    """
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ] or isinstance(element, bs4.element.Comment):
        return False
    return True


def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
    """Scrapes a URL for all text information.

    Args:
        url: URL of webpage to scrape.
        timeout: Timeout of the requests call.
    Returns:
        web_text: The visible text of the scraped URL.
        url: URL input.
    """
    # Scrape the URL
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as _:
        return None, url

    # Extract out all text from the tags
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        texts = soup.findAll(string=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:
        return None, url

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url


def search_bing(query: str, timeout: float = 3, sub_key: str = None) -> List[str]:
    """Searches the query using Bing.
    Args:
        query: Search query.
        timeout: Timeout of the requests call.
    Returns:
        search_results: A list of the top URLs relevant to the query.
    """

    endpoint = SEARCH_URL
    headers = {"Ocp-Apim-Subscription-Key": sub_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        response = response.json()
        search_results = [r["url"] for r in response["webPages"]["value"]]
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        search_results = None
    except Exception as e:
        print(f"An error occurred: {e}")
        search_results = None
    
    # print(f"Query: {query}\nSearch Results: {search_results}")
    return search_results


def run_search(
    query: str,
    cached_search_results: List[str] = None,
    max_search_results_per_query: int = 3,
    max_sentences_per_passage: int = 5,
    sliding_distance: int = 1,
    max_passages_per_search_result_to_return: int = 1,
    timeout: float = 3,
    randomize_num_sentences: bool = False,
    filter_sentence_len: int = 250,
    max_passages_per_search_result_to_score: int = None,
    sub_key: str = None,
    ranking_model: str = None
) -> List[Dict[str, Any]]:
    """Searches the query on a search engine and returns the most relevant information.

    Args:
        query: Search query.
        max_search_results_per_query: Maximum number of search results to get return.
        max_sentences_per_passage: Maximum number of sentences for each passage.
        filter_sentence_len: Maximum length of a sentence before being filtered.
        sliding_distance: Sliding distance over the sentences of each search result.
            Used to extract passages.
        max_passages_per_search_result_to_score: Maxinum number of passages to score for
            each search result.
        max_passages_per_search_result_to_return: Maximum number of passages to return
            for each search result.
    Returns:
        retrieved_passages: Top retrieved passages for the search query.
    """
    if cached_search_results is not None:
        search_results = cached_search_results
    else:
        search_results = search_bing(query, timeout=timeout, sub_key=sub_key)

    # Scrape search results in parallel
    with concurrent.futures.ThreadPoolExecutor() as e:
        scraped_results = e.map(scrape_url, search_results, itertools.repeat(timeout))
    # Remove URLs if we weren't able to scrape anything or if they are a PDF.
    scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]

    # Iterate through the scraped results and extract out the most useful passages.
    retrieved_passages = []
    for webtext, url in scraped_results[:max_search_results_per_query]:
        if randomize_num_sentences:
            sents_per_passage = random.randint(1, max_sentences_per_passage)
        else:
            sents_per_passage = max_sentences_per_passage

        # Chunk the extracted text into passages.
        passages = chunk_text(
            text=webtext,
            sentences_per_passage=sents_per_passage,
            filter_sentence_len=filter_sentence_len,
            sliding_distance=sliding_distance,
        )
        passages.append("dummy")
        passages = passages[:max_passages_per_search_result_to_score]
        if not passages:
            continue

        # Score the passages by relevance to the query using a cross-encoder.
        if ranking_model == "cross_encoder":
            scores = PASSAGE_RANKER.predict([(query, p) for p in passages]).tolist()
        # elif ranking_model == "cohere":
        #     scores = cohere_call.get_passage_ranking_score(query, passages)
           
        passage_scores = list(zip(passages, scores))

        # Take the top passages_per_search passages for the current search result.
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
            retrieved_passages.append(
                {
                    "text": passage,
                    "url": url,
                    "query": query,
                    "sents_per_passage": sents_per_passage,
                    "retrieval_score": score,  # Cross-encoder score as retr score
                }
            )

    if retrieved_passages:
        # Sort all retrieved passages by the retrieval score.
        retrieved_passages = sorted(
            retrieved_passages, key=lambda d: d["retrieval_score"], reverse=True
        )

        # Normalize the retreival scores into probabilities
        scores = [r["retrieval_score"] for r in retrieved_passages]
        probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
        for prob, passage in zip(probs, retrieved_passages):
            passage["score"] = prob

    return retrieved_passages

if __name__ == "__main__":

    max_search_results_per_query = 5
    max_sentences_per_passage = 4
    sliding_distance = 1
    max_passages_per_search_result = 1
    max_evidences_per_question = 1

    target_sent = "Shah Rukh Khan, often referred to as the King of Bollywood, is an Indian actor and film producer primarily working in Hindi cinema, recognized for his charm and romantic roles."

    questions = ['For which film industry does Shah Rukh Khan primarily work in Maharashtra, India?', 
    'In which Indian state does Shah Rukh Khan primarily work for his Bollywood career?', 
    "What are some notable attributes associated with Shah Rukh Khan's career in Bollywood?", 
    "What is the primary focus of Shah Rukh Khan's work in Hindi cinema in Maharashtra?", 
    'What nickname is Shah Rukh Khan often referred to in the Indian film industry?']

    # Run search on generated question for the claim
    t1 = time.time()
    evidences_for_questions = [
        run_search(
            query=query,
            max_search_results_per_query=max_search_results_per_query,
            max_sentences_per_passage=max_sentences_per_passage,
            sliding_distance=sliding_distance,
            max_passages_per_search_result_to_return=max_passages_per_search_result,
            sub_key = api.SUBSCRIPTION_KEY,
            max_passages_per_search_result_to_score=30,
            ranking_model="cohere"
        )
        for query in questions
    ]
    t2 = time.time()
    print(f"Time taken: {(t2-t1)/60:.2f} mint")
    # Flatten the evidences per question into a single list.
    used_evidences = [
        e
        for cur_evids in evidences_for_questions
        for e in cur_evids[:max_evidences_per_question]
    ]

    output = {
        "claim_target": target_sent,
        "questions": questions,
        "evidences": used_evidences
    }

    print(output)