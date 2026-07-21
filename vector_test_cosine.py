import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime, timedelta
from basic_email import get_emails_lang

import re
from rank_bm25 import BM25Okapi

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

def email_sort_storing(emails, collection =None):
    ids = []
    valid_docs = []
    for email in emails:
        # 1. Skip emails that have no text content to avoid embedding errors
        email_text = email.get("text", "").strip()
        if not email_text:
            continue
        # 2. Add to collection wrapping everything in a LIST []
        date_format = "%b %d, %Y %H:%M"
        try:
            datetime_answer = (datetime.strptime(email.get("date"), date_format))
            time = datetime.strftime(datetime_answer, "%H:%M")
            date =datetime.strftime(datetime_answer, "%d/%m/%Y")
        except(ValueError, TypeError):
            date , time = "unknown_Date", "unknown_time"
        unique_id = f"{email.get('subject', 'unknown_subject')} - {email.get("sender")} - {str(date)}"
        ids.append(unique_id)
        valid_docs.append({"id": unique_id, "text": email_text})
        if collection:
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
    return collection, ids, valid_docs

def vector_search(emails):
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
        )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}, #"hnsw:space": "ip" for dot product but need different model
        )
    collection, _ = email_sort_storing(emails, collection)
    query_results = collection.query(
        query_texts=["Any emails about american express?"],
        n_results=5,
        # where={"date":"05/07/2026"}
    )
    # print(query_results.keys())
    # print(query_results["ids"])
    # print(query_results["ids"])
    # print(query_results["distances"])
    # print(query_results["metadatas"])
    return query_results["ids"][0]

class LocalBM25Search:
    def __init__(self, valid_documents):
        """
        documents format: [{"id": "email_1", "text": "hello world"}, ...]
        """
        self.doc_ids = [doc["id"] for doc in valid_documents]
        self.raw_texts = [doc["text"] for doc in valid_documents]
        
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
collection = vector_search(emails)
collection, ids, valid_docs = email_sort_storing(emails=emails, collection=collection) 
returned_results = 5
query_results = collection.query(
    query_texts=["Any emails about american express?"],
    n_results=returned_results,
    # where={"date":"05/07/2026"}
)
bm25 = LocalBM25Search(valid_docs)
matching_ids = bm25.search("american", top_k=returned_results)


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for system_ranking in rankings:
        for rank, doc_id in enumerate(system_ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


combined = reciprocal_rank_fusion([matching_ids, query_results["ids"][0]])
print(combined)
final_ids = [doc_id for doc_id, score in combined]
print(final_ids)