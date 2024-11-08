import numpy as np
from db import store_embedding, query_embedding, get_embedding_from_model

# Sample historical places and metadata
historical_places = [
    ("The Colosseum in Rome, Italy, is an ancient amphitheater where gladiators once fought.", "1"),
    ("The Great Wall of China was built to protect against invasions and stretches across northern China.", "2"),
    ("Machu Picchu in Peru is an ancient Incan city set high in the Andes Mountains.", "3"),
    ("The Pyramids of Giza in Egypt are one of the Seven Wonders of the Ancient World.", "4"),
    ("The Taj Mahal in India is a white marble mausoleum built by Emperor Shah Jahan in memory of his wife Mumtaz Mahal.", "5"),
    ("The Eiffel Tower in Paris, France, is an iconic wrought-iron lattice tower that was originally a temporary exhibit for the 1889 World's Fair.", "6"),
    ("The Acropolis of Athens in Greece is an ancient citadel with ruins like the Parthenon, a symbol of classical civilization.", "7"),
    ("The Machu Picchu Inca Trail in Peru leads to the ancient city of Machu Picchu, perched on the Andes Mountains.", "8"),
    ("The Alhambra in Granada, Spain, is a Moorish palace and fortress complex renowned for its stunning Islamic architecture and beautiful gardens.", "9")
]


# Store embeddings for historical places in Pinecone
for text, doc_id in historical_places:
    embedding = get_embedding_from_model(text)  
    metadata = {"text": text}
    store_embedding(doc_id=doc_id, embedding=embedding, metadata=metadata)

print("Historical places' embeddings stored in Pinecone.")

query_text = input("Enter your query: ").lower()  

# Get the query embedding (converted to vector)
query_vector = get_embedding_from_model(query_text)

# Query Pinecone with the query embedding
results = query_embedding(query_vector, top_k=3)

# Display the query results (only relevant results based on query)
print("Query Results:")
for match in results['matches']:
    # Check if any keyword from the query is in the text of the matched document
    if any(keyword in match['metadata']['text'].lower() for keyword in query_text.split()):
        print(f"Text: {match['metadata']['text']}")
