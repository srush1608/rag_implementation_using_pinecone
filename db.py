import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = 'srushti11'
index = pc.Index(index_name)

# Function to store embeddings
def store_embedding(doc_id, embedding, metadata=None):
    """Store embedding and associated document in Pinecone."""
    metadata = metadata or {}
    index.upsert(vectors=[{"id": doc_id, "values": embedding.tolist(), "metadata": metadata}])

# Function to query embeddings
def query_embedding(query_embedding, top_k=3):
    """Query the Pinecone index using the provided query embedding."""
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    return results

# Function to generate embeddings from a model
def get_embedding_from_model(text):
    """Generate an embedding for the given text using a model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model, replace with your own model if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input and get embeddings
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # No need to track gradients
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Get the average of the last hidden state
    return embeddings[0]  
