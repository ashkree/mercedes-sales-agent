# app/query_interface.py

import faiss
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers package not installed.")
    print("Please install with: pip install sentence-transformers")
    sys.exit(1)

# Use absolute paths based on the project root
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss" / "index.bin"
ID_MAP_PATH = BASE_DIR / "data" / "faiss" / "id_map.json"
DOCSTORE_PATH = BASE_DIR / "data" / "raw" / "docstore.jsonl"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Initialize model
model = None


def load_model():
    """Load the sentence transformer model"""
    global model
    if model is None:
        try:
            model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model


def ensure_index_exists():
    """Ensure the FAISS index is initialized before querying"""
    if not FAISS_INDEX_PATH.exists() or not ID_MAP_PATH.exists():
        print("FAISS index not found. Creating from docstore...")
        try:
            # Import here to avoid circular imports
            from scripts.embed_chunks import main as embed_main
            embed_main()
        except Exception as e:
            print(f"Error creating index: {e}")
            print("Using fallback search method")
            return False
    return True


def load_docstore():
    """Load the document store with extracted metadata"""
    if not DOCSTORE_PATH.exists():
        return {}

    docstore = {}
    with open(DOCSTORE_PATH) as f:
        for line in f:
            doc = json.loads(line)

            # Extract metadata from text format "Key: Value | Key2: Value2"
            metadata = {}
            if "text" in doc:
                text_parts = doc["text"].split(" | ")
                for part in text_parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        metadata[key.strip().lower()] = value.strip()

            # Add metadata to doc
            doc["metadata"] = metadata
            docstore[doc['id']] = doc

    return docstore


def fallback_search(query, top_k=3):
    """Simple keyword-based search as fallback"""
    if not DOCSTORE_PATH.exists():
        return ["No data found. Please run setup.py first."]

    # Load docstore
    documents = []
    with open(DOCSTORE_PATH) as f:
        for line in f:
            documents.append(json.loads(line))

    # Simple keyword matching
    query_terms = query.lower().split()
    results = []

    for doc in documents:
        text = doc["text"].lower()
        score = sum(1 for term in query_terms if term in text)
        if score > 0:
            results.append((doc, score))

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_k]]


def search(query, top_k=5, return_metadata=False):
    """
    Search for the most relevant documents

    Args:
        query (str): The search query
        top_k (int): Number of results to return
        return_metadata (bool): If True, return full document objects including metadata

    Returns:
        list: Either list of strings (texts) or list of dicts (full documents)
    """
    try:
        # Check if index exists
        if not ensure_index_exists():
            results = fallback_search(query, top_k)
            return results if return_metadata else [r["text"] for r in results]

        # Load model
        model = load_model()
        if model is None:
            results = fallback_search(query, top_k)
            return results if return_metadata else [r["text"] for r in results]

        # Load index
        index = faiss.read_index(str(FAISS_INDEX_PATH))

        # Load ID map
        with open(ID_MAP_PATH) as f:
            id_map = json.load(f)

        # Load docstore
        docstore = load_docstore()

        # Encode query
        q_vec = model.encode([query])

        # Search
        D, I = index.search(np.array(q_vec), top_k)

        # Get results
        results = []
        for i in I[0]:
            if str(i) in id_map and id_map[str(i)] in docstore:
                results.append(docstore[id_map[str(i)]])

        if not return_metadata:
            results = [r["text"] for r in results]

        return results
    except Exception as e:
        print(f"Error during search: {e}")
        results = fallback_search(query, top_k)
        return results if return_metadata else [r["text"] for r in results]


def get_model_by_name(model_name):
    """Get a specific Mercedes model by name"""
    docstore = load_docstore()

    model_name_lower = model_name.lower()

    # Look for model by name
    for doc_id, doc in docstore.items():
        if "metadata" in doc and "model" in doc["metadata"]:
            doc_model = doc["metadata"]["model"].lower()
            if model_name_lower in doc_model:
                return doc

    return None


def get_models_by_criteria(criteria):
    """
    Get models matching specific criteria

    Args:
        criteria (dict): Dictionary of criteria to match (e.g., {"body style": "SUV"})

    Returns:
        list: List of matching models
    """
    docstore = load_docstore()
    results = []

    for doc_id, doc in docstore.items():
        if "metadata" not in doc:
            continue

        match = True
        for key, value in criteria.items():
            key_lower = key.lower()
            if key_lower not in doc["metadata"] or value.lower() not in doc["metadata"][key_lower].lower():
                match = False
                break

        if match:
            results.append(doc)

    return results


if __name__ == "__main__":
    while True:
        q = input("❓ Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        results = search(q)
        print("\n📄 Top Results:")
        for res in results:
            print("-", res)
        print()
