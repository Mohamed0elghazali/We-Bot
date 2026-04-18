### load from disk
from langchain_community.vectorstores import FAISS

from .clients import embeddings
from .utils import timeit

vector_db = FAISS.load_local("vectordb", embeddings, allow_dangerous_deserialization=True)

import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain.tools import tool

@tool
@timeit
def search_kb(query: str, k_results: int = 5, score_threshold: float = 0.1) -> List[Document]:    
    """
    Search the internal Knowledge Base to retrieve relevant text chunks for a given query.
    
    This tool performs a semantic similarity search against a vector database. It is 
    best used when the user asks specific questions about internal documentation, 
    stored procedures, or historical data that requires factual context.

    Args:
        query (str): The natural language question or search terms provided by the user.

    Returns:
        List[Document]: A list of LangChain Document objects. Each document contains 
            the text 'page_content' and 'metadata' including the 'similarity_score'.
            If no documents meet the threshold, an empty list is returned.
    """
    results = vector_db.similarity_search_with_score(query, k=k_results)
    
    docs_with_scores = []
    for doc, distance in results:
        # Normalize distance to 0-1 similarity (1 is best)
        # similarity = float(np.exp(-distance))
        similarity = float(1/(1+distance))
        
        # Now your threshold makes more sense (e.g., 0.4 similarity)
        if similarity >= score_threshold:
            doc.metadata["similarity_score"] = round(similarity, 4)
            docs_with_scores.append(doc)
            
    return docs_with_scores