import time
import torch
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load embedding model once
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # Consider exiting or handling the error gracefully
    model = None

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

@app.post("/embed")
def embed(request: EmbedRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Embedding model is not loaded.")
        
    try:
        # Time the model's encode method specifically
        start_time = time.perf_counter()
        embeddings = model.encode(request.texts, normalize_embeddings=request.normalize)
        end_time = time.perf_counter()
        
        # Log the internal timing
        logging.info(f"Encoded {len(request.texts)} sentences in {end_time - start_time:.3f} seconds.")

        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("embedding_server:app", host="0.0.0.0", port=9096, reload=True)
