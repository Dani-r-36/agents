import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)



embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
    )

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}, #"hnsw:space": "ip" for dot product but need different model
    )

>>> documents = [
...     "The latest iPhone model comes with impressive features and a powerful camera.",
...     "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
...     "Einstein's theory of relativity revolutionized our understanding of space and time.",
...     "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
...     "The American Revolution had a profound impact on the birth of the United States as a nation.",
...     "Regular exercise and a balanced diet are essential for maintaining good physical health.",
...     "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
...     "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
...     "Startup companies often face challenges in securing funding and scaling their operations.",
...     "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
... ]

>>> genres = [
...     "technology",
...     "travel",
...     "science",
...     "food",
...     "history",
...     "fitness",
...     "art",
...     "climate change",
...     "business",
...     "music",
... ]

>>> collection.add(
...     documents=documents,
...     ids=[f"id{i}" for i in range(len(documents))],
...     metadatas=[{"genre": g} for g in genres]
... )



emails = get_emails_lang((datetime.datetime.now() - datetime.timedelta(days=31)).strftime('%Y/%m/%d'))
new_docs = []

for email in emails:
    # 1. Skip emails that have no text content to avoid embedding errors
    email_text = email.get("text", "").strip()
    if not email_text:
        continue
        
    # 2. Add to collection wrapping everything in a LIST []
    collection.add(
        documents=[email_text],
        ids=[email.get("subject", "unknown_subject")],  # Essential fix for the syntax error
        metadatas=[{
            "source": "email",
            "date": email.get("date"),
            "sender": email.get("sender"),
            "subject": email.get("subject")
        }]
    )

collection.add(
    documents=new_docs,
    ids=[f"id{i}" for i in range(len(new_docs))],
    metadatas=[{"genre": g} for g in genres]
)
