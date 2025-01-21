from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    """
    Expects a str as argument and returns embedding of the sentence as a torch.tensor()
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

class ArticleTitles(BaseModel):
    reference: str
    other: List[str]

# Access UI on http://127.0.0.1:8000/docs
@app.post("/compare-titles/")
async def compare_titles(titles: ArticleTitles):
    """
    Input: JSON object as defined by ArticleTitles
    Example input:
        {   “reference”: “Higgs boson in particle physics”, 
            “other”: [“Best soup recipes”, “Basel activities”, “Particle physics at CERN”]
        }
    """
    reference = titles.reference
    other = titles.other

    # Compute embeddings
    reference_embedding = get_embedding(reference)
    other_embeddings = [get_embedding(title) for title in other]

    # Compute similarity and return most similar
    cos = nn.CosineSimilarity(dim=1)
    similarities = []
    for embedding in other_embeddings:
        similarity = cos(reference_embedding, embedding)
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)
    top_result = other[most_similar_index]

    return { "top_result": top_result }
