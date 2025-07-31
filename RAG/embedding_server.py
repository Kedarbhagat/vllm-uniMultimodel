from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import uvicorn
import torch

# Load embedding model once

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

@app.post("/embed")
def embed(request: EmbedRequest):
    try:
        embeddings = model.encode(request.texts, normalize_embeddings=request.normalize)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: python embedding_server.py
if __name__ == "__main__":
    uvicorn.run("embedding_server:app", host="0.0.0.0", port=9096)
