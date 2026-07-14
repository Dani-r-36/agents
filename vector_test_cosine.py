import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime, timedelta
from basic_email import get_emails_lang

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"


def vector_search():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
        )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}, #"hnsw:space": "ip" for dot product but need different model
        )
    emails = get_emails_lang((datetime.now() - timedelta(days=14)).strftime('%Y/%m/%d'))
    for email in emails:
        # 1. Skip emails that have no text content to avoid embedding errors
        email_text = email.get("text", "").strip()
        if not email_text:
            continue
        # 2. Add to collection wrapping everything in a LIST []
        date_format = "%b %d, %Y %H:%M"
        datetime_answer = (datetime.strptime(email.get("date"), date_format))
        time = datetime.strftime(datetime_answer, "%H:%M")
        date =datetime.strftime(datetime_answer, "%d/%m/%Y")
        unique_id = f"{email.get('subject', 'unknown_subject')} - {email.get("sender")} - {str(date)}"
        # print(date)
        collection.add(
            documents=[email_text],
            ids=[unique_id],  # Essential fix for the syntax error
            metadatas=[{
                "source": "email",
                "date": date,
                "time": time,
                "sender": email.get("sender"),
                "subject": email.get("subject")
            }]
        )
    query_results = collection.query(
        query_texts=["What's my emails this day?", "what is the most important email I have "],
        n_results=10,
        where={"date":"05/07/2026"}
    )
    # print(query_results.keys())
    print(query_results["ids"])
    # print(query_results["ids"])
    print(query_results["distances"])
    # print(query_results["metadatas"])
    return query_results


import re
from rank_bm25 import BM25Okapi

class LocalBM25Search:
    def __init__(self, documents, email_id):
        """
        documents format: [{"id": "email_1", "text": "hello world"}, ...]
        """
        self.doc_ids = email_id
        self.raw_texts = [doc["text"] for doc in documents]
        
        # Build tokenized corpus
        tokenized_corpus = [self._clean_and_tokenize(text) for text in self.raw_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def _clean_and_tokenize(self, text: str) -> list[str]:
        # Simple cleanup: lowercase and strip out weird punctuation
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return cleaned.split()

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Returns the top_k Document IDs"""
        tokenized_query = self._clean_and_tokenize(query)
        
        # Get top-scoring document IDs
        top_ids = self.bm25.get_top_n(tokenized_query, self.doc_ids, n=top_k)
        return top_ids
    

emails = get_emails_lang((datetime.now() - timedelta(days=14)).strftime('%Y/%m/%d'))
date_format = "%b %d, %Y %H:%M"
ids = [f"{email.get('subject', 'unknown_subject')} - {email.get("sender")} - {str(datetime.strftime(datetime.strptime(email.get("date"), date_format)), "%d/%m/%Y")}" for email in emails]
bm25 = LocalBM25Search(emails, ids)