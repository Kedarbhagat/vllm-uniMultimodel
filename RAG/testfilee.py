# Save this as 'test_embedding_speed.py'
import time
from sentence_transformers import SentenceTransformer
import torch

def test_model_speed():
    """
    Measures the speed of the embedding model without any server overhead.
    """
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Time model loading
    print("Loading model...")
    start_load_time = time.perf_counter()
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        end_load_time = time.perf_counter()
        print(f"Model loaded successfully in {end_load_time - start_load_time:.3f} seconds.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare dummy text for encoding
    texts_to_embed = ["This is a test sentence.", "Another sentence to check speed."]
    
    # Time the encoding process
    print("\nStarting encoding test...")
    start_encode_time = time.perf_counter()
    try:
        model.encode(texts_to_embed, normalize_embeddings=True)
        end_encode_time = time.perf_counter()
        print(f"Encoding {len(texts_to_embed)} sentences took {end_encode_time - start_encode_time:.3f} seconds.")
    except Exception as e:
        print(f"Error during encoding: {e}")
        return

if __name__ == "__main__":
    test_model_speed()
