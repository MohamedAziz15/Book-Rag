import getpass
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

import logging

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_pdf(file_path: str):
    # file_path = "docs/Mohamed-Aziz-AI Engineer-.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    logging.debug("loading pdf metadata:")
    print(docs[0].metadata)
    return docs



def split_docs(docs):
    #RecursiveCharacterTextSplitter: recursively split the document using common separators like new lines until each chunk is the appropriate size.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters) overlap helps mitigate the possibility of separating a statement from important context related to it
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")
    return all_splits



if __name__ == "__main__":
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary)
    )
    docs = load_pdf("docs/Mohamed-Aziz-AI Engineer-.pdf")
    split_docs = split_docs(docs)
    vector_store.add_documents(split_docs)
    logging.info("Document added to vector store successfully.")




# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )



# from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# persist_directory = 'docs/chroma/'
# embedding = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)




