from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS

loader = PyPDFLoader("attention.pdf")
documents = loader.load()

ollama_emb = OllamaEmbeddings()

db = FAISS.from_documents(documents,ollama_emb)
print(db)