import numpy as np
from db import store_embedding, query_embedding, get_embedding_from_model, extract_text_from_pdf, chunk_text

# Load and process the PDF document
pdf_path = "documents.pdf"  # Your PDF file
pdf_text = extract_text_from_pdf(pdf_path)  # Extract text from PDF

# Chunk the PDF text into smaller parts
chunks = chunk_text(pdf_text)

# Generate embeddings for each chunk and store in Pinecone
for idx, chunk in enumerate(chunks):
    embedding = get_embedding_from_model(chunk)  # Generate the embedding for the chunk
    doc_id = str(idx)  # Use the index as a document ID
    store_embedding(doc_id=doc_id, embedding=embedding, metadata={"text": chunk})

print("Document embeddings stored in Pinecone.")

# Query for a specific text from the document
query_text = input("Enter your query: ").lower()  
query_vector = get_embedding_from_model(query_text)  # Convert query to embedding

# Query Pinecone for top matches
results = query_embedding(query_vector, top_k=3)

# Display the retrieved results (documents or chunks of the large document)
print("Query Results:")
for match in results['matches']:
    print(f"Text: {match['metadata']['text']}")
