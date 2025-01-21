from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import SentenceTransformer

app = FastAPI()

class ArticleTitles(BaseModel):
    reference: str
    other: List[str]

# run uvicorn main:app --reload
# and ccess UI at http://127.0.0.1:8000/docs
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

    # compute embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [reference] + other
    embeddings = model.encode(sentences)

    # compute similarity and return top result
    # by default, this uses cosine: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity
    similarities = model.similarity(embeddings, embeddings)[1:, 0]

    most_similar_index = torch.argmax(similarities) + 1
    top_result = sentences[most_similar_index]

    return { "top_result": top_result }
