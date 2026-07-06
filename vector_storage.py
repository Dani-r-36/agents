import datetime
import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from basic_email import get_emails, get_emails_lang

st.set_page_config(layout="wide")
main_content, log_content = st.columns(2)

@st.cache_resource
def get_vector_store():
    """
    Initializes a completely local embedding model and links it to 
    the in-memory vector store.
    """
    # 'all-MiniLM-L6-v2' is extremely fast, accurate, and lightweight (around 90MB)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return InMemoryVectorStore(embeddings)

vector_store = get_vector_store()

# 2. Create a clean, UN-CACHED function to inject new data on the fly
def store_new_emails(email_list: list):
    """Converts raw emails into LangChain Documents and streams them into the live store."""
    if not email_list:
        return
        
    new_docs = []
    for email in email_list:
        doc = Document(
            page_content=email.get("text", ""),
            metadata={
                "source": "email",
                "date": email.get("date"),
                "sender": email.get("sender"),
                "subject": email.get("subject")
            }
        )
        new_docs.append(doc)
    
    # This instantly updates the live vector store in RAM
    vector_store.add_documents(new_docs)

# Add this to the bottom of your log_content column
# with log_content.expander("🔍 View Live Vector Store Contents"):
#     # Access the internal dictionary
#     raw_store = vector_store.store
    
#     if not raw_store:
#         st.info("Vector store is currently empty.")
#     else:
#         st.success(f"Database contains {len(raw_store)} document chunks.")
        
#         # Format the internal dictionary into something clean for st.json
#         readable_database = {}
#         for idx, (doc_id, doc) in enumerate(raw_store.items()):
#             readable_database[f"Chunk {idx+1} (ID: {doc_id[:8]})"] = {
#                 "metadata": doc.metadata,
#                 "content_preview": doc.page_content[:150] + "..."
#             }
            
#         # This will render an interactive, expandable JSON tree in your UI
#         st.json(readable_database)
emails = get_emails_lang((datetime.datetime.now() - datetime.timedelta(days=31)).strftime('%Y/%m/%d'))
# print("emails ", emails)
store_new_emails(emails)
total_docs = len(vector_store.store)
print(f"Total documents in vector store: {total_docs}")

# 2. Loop through and view the stored documents and their metadata
for doc_id, doc in vector_store.store.items():
    print(f"--- Document ID: {doc_id} ---")
    
    # Since 'doc' is a dictionary, we use .get() to access keys safely
    metadata = doc.get("metadata", {})
    
    print(f"Subject: {metadata.get('subject')}")
    print(f"Sender: {metadata.get('sender')}")
    
    # You can also grab the content preview safely:
    # print(f"Content Preview: {doc.get('page_content', '')[:100]}...")