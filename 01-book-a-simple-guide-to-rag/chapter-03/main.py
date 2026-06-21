from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def load_web_page(url: str):
    """
    Load a web page and return its content as a list of documents.
    
    Args:
        url (str): The URL of the web page to load.

    Returns:
        list: A list of documents containing the web page content.
    """
    loader = AsyncHtmlLoader(url)
    documents = loader.load()
    return documents

def clean_data(documents):
    """
    Clean the loaded documents by removing unnecessary whitespace and formatting.

    Args:
        documents (list): A list of documents to clean.
    
    Returns:
        list: A list of cleaned documents.
    """
    html2text_transformer = Html2TextTransformer()

    return html2text_transformer.transform_documents(documents)

def split_documents(documents, chunk_size=1000, chunk_overlap=200, separator="\n"):
    """
    Split the cleaned documents into smaller chunks for easier processing.

    Args:
        documents (list): A list of cleaned documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
        separator (str): The separator to use when splitting the documents.
    
    Returns:
        list: A list of document chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    return text_splitter.split_documents(documents)

def create_embedding_model():
    """
    Create an OpenAI embedding model for document chunks.
    
    Returns:
        OpenAIEmbeddings: An embedding model for the document chunks.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

def create_vectorstore(chunks, embedding_model):
    """
    Create a vector store from the document chunks.

    Args:
        chunks (list): A list of document chunks.
        embedding_model (OpenAIEmbeddings): The embedding model to use.
    
    Returns:
        FAISS: A FAISS vector store containing the document chunks and their embeddings.
    """
    return FAISS.from_documents(chunks, embedding_model)

def main():
    url = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"  # Replace with the desired URL
    documents = load_web_page(url)
    cleaned_documents = clean_data(documents)
    print(f"Loaded {len(cleaned_documents)} documents from {url}")
    print("\nMetadata of the first document:")
    print(cleaned_documents[0].metadata)
    print("\nContent of the first document (first 500 characters):")
    print(cleaned_documents[0].page_content[:500])

    chunks = split_documents(cleaned_documents)
    print(f"\nSplit into {len(chunks)} chunks.")
    print("\nContent of the 4 chunk (last 200 characters):")
    print(chunks[3].page_content[-200:])
    print("\nContent of the 5 chunk (first 200 characters):")
    print(chunks[4].page_content[:200])

    embedding_model = create_embedding_model()

    vectorstore = create_vectorstore(chunks, embedding_model)
    print(f"\nCreated vector store with {vectorstore.index.ntotal} chunks.")

    vectorstore.save_local("faiss_index")
    query = "Who won the 2023 Cricket World Cup?"
    docs = vectorstore.similarity_search(query)
    print(f"\nFound {len(docs)} documents similar to the query: '{query}'")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i + 1} (first 200 characters):")
        print(doc.page_content[:200])

if __name__ == "__main__":
    main()
