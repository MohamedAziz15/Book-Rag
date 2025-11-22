import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from gemini_llm import gemini_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookRAGSystem:
    """Manages multiple books in a RAG system with separate vector stores."""
    
    def __init__(self, persist_directory: str = "./chroma_langchain_db"):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.books = {}  # book_id -> book_info
        self.vector_stores = {}  # book_id -> vector_store
        self._load_existing_books()
    
    def _get_collection_name(self, book_id: str) -> str:
        """Generate a collection name for a book."""
        return f"book_{book_id}"
    
    def _load_existing_books(self):
        """Load existing books from the vector store."""
        try:
            # Try to get list of collections from Chroma
            if os.path.exists(self.persist_directory):
                # List all collection directories
                db_path = Path(self.persist_directory)
                if db_path.exists():
                    # Chroma stores collections in the database
                    # We'll track books separately in a metadata file
                    metadata_file = Path(self.persist_directory) / "books_metadata.json"
                    if metadata_file.exists():
                        import json
                        with open(metadata_file, 'r') as f:
                            self.books = json.load(f)
                            # Initialize vector stores for existing books
                            for book_id in self.books.keys():
                                self._get_vector_store(book_id)
        except Exception as e:
            logger.warning(f"Could not load existing books: {e}")
    
    def _save_books_metadata(self):
        """Save books metadata to file."""
        metadata_file = Path(self.persist_directory) / "books_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(self.books, f, indent=2)
    
    def _get_vector_store(self, book_id: str) -> Chroma:
        """Get or create vector store for a book."""
        if book_id not in self.vector_stores:
            collection_name = self._get_collection_name(book_id)
            self.vector_stores[book_id] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self.vector_stores[book_id]
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from PDF")
        return docs
    
    def split_docs(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        logger.info(f"Split document into {len(all_splits)} chunks")
        return all_splits
    
    def add_book(self, file_path: str, book_name: Optional[str] = None) -> str:
        """Add a new book to the system."""
        # Generate book ID from file path and name
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        book_id = f"{Path(file_path).stem}_{file_hash}"
        
        if book_id in self.books:
            logger.warning(f"Book {book_id} already exists")
            return book_id
        
        # Load and process the book
        logger.info(f"Loading book from {file_path}")
        docs = self.load_pdf(file_path)
        all_splits = self.split_docs(docs)
        
        # Get or create vector store for this book
        vector_store = self._get_vector_store(book_id)
        
        # Add documents to vector store
        # Chroma automatically persists when persist_directory is set
        vector_store.add_documents(all_splits)
        
        # Store book metadata
        book_name = book_name or Path(file_path).stem
        self.books[book_id] = {
            "name": book_name,
            "file_path": file_path,
            "chunks": len(all_splits),
            "pages": len(docs)
        }
        self._save_books_metadata()
        
        logger.info(f"Book '{book_name}' added successfully with ID: {book_id}")
        return book_id
    
    def get_books(self) -> dict:
        """Get list of all books."""
        return self.books
    
    def delete_book(self, book_id: str) -> bool:
        """Delete a book from the system."""
        if book_id not in self.books:
            return False
        
        try:
            # Delete vector store collection using Chroma client
            vector_store = self._get_vector_store(book_id)
            collection_name = self._get_collection_name(book_id)
            
            # Get all document IDs and delete them
            try:
                # Try to get all ids from the collection
                all_ids = vector_store.get()['ids'] if hasattr(vector_store, 'get') else []
                if all_ids:
                    vector_store.delete(ids=all_ids)
            except Exception as e:
                logger.warning(f"Could not delete documents directly: {e}")
            
            # Delete collection using Chroma client
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_directory)
                client.delete_collection(name=collection_name)
            except Exception as e:
                logger.warning(f"Could not delete collection via client: {e}")
            
            # Remove from memory
            if book_id in self.vector_stores:
                del self.vector_stores[book_id]
            del self.books[book_id]
            self._save_books_metadata()
            
            logger.info(f"Book {book_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting book {book_id}: {e}")
            return False
    
    def get_retriever(self, book_id: str, k: int = 4):
        """Get retriever for a specific book."""
        if book_id not in self.books:
            raise ValueError(f"Book {book_id} not found")
        
        vector_store = self._get_vector_store(book_id)
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    def query_book(self, book_id: str, question: str, k: int = 4) -> str:
        """Query a specific book using RAG."""
        if book_id not in self.books:
            return f"Error: Book not found. Please select a valid book."
        
        try:
            retriever = self.get_retriever(book_id, k=k)
            
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context from a book. "
                "Use the given context to answer the question accurately and comprehensively. "
                "If the context doesn't contain enough information to answer the question, "
                "say that you don't have enough information from the book to answer. "
                "Keep your answer clear and well-structured. "
                "Context from the book: {context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(gemini_llm, prompt)
            chain = create_retrieval_chain(retriever, question_answer_chain)
            
            result = chain.invoke({"input": question})
            return result.get("answer", "Sorry, I couldn't generate an answer.")
            
        except Exception as e:
            logger.error(f"Error querying book: {e}")
            return f"Error: {str(e)}"


# Global instance
_rag_system = None

def get_rag_system() -> BookRAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = BookRAGSystem()
    return _rag_system





