import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text: str) -> list:
   
    # Convert to lowercase and extract alphanumeric tokens
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    
    stop_words = {
        'the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
    }
    
    # Filter out stop words and single characters
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    return tokens

def compute_cosine_similarity(text1: str, text2: str) -> float:
    # Handle empty inputs
    if not text1.strip() or not text2.strip():
        return 0.0
    
    try:
        # Create vectorizer with custom tokenizer
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize, 
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1
        )
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(sim_score)
    except Exception:
        return 0.0
