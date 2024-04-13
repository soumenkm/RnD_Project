# This snippet shows and example how to use the Cohere Embed V3 models for semantic search.
# Make sure to have the Cohere SDK in at least v4.30 install: pip install -U cohere 
# Get your API key from: www.cohere.com
import cohere, sys
sys.path.append("/root/RnD_Project/utils")
import numpy as np
import api

def get_passage_ranking_score(query, docs):

    cohere_key = api.COHERE_KEY   #Get your API key from www.cohere.com
    co = cohere.Client(cohere_key)

    #Encode your documents with input type 'search_document'
    doc_emb = co.embed(texts=docs, input_type="search_document", model="embed-english-v3.0").embeddings
    doc_emb = np.asarray(doc_emb)

    #Encode your query with input type 'search_query'
    query_emb = co.embed(texts=[query], input_type="search_query", model="embed-english-v3.0").embeddings
    query_emb = np.asarray(query_emb)
    query_emb.shape

    #Compute the dot product between query embedding and document embedding
    scores = np.dot(query_emb, doc_emb.T)[0]
    
    return scores

if __name__ == "__main__":
    query = "What is PyTorch?"
    docs = ["Pytorch is Machine Learning library", 
            "Pytorch is Deep Learning library", 
            "Pytoch is Reinforcement Learning library"]
    scores = get_passage_ranking_score(query, docs)
    print(scores)