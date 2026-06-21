from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load vector store from local directory
def load_vectorstore():
    """
    Load a vector store from a local directory.

    Returns:
        FAISS: A loaded FAISS vector store.
    """
    return FAISS.load_local(
        folder_path="/Users/harisubramaniam/learning/azure-ai/rag-learning-lab/01-book-a-simple-guide-to-rag/chapter-03/faiss_index/",
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key),
        allow_dangerous_deserialization=True,
    )

def main():
    vectorstore = load_vectorstore()
    print(f"\nLoaded vector store with {vectorstore.index.ntotal} chunks.")
    
    query = "Who won the 2023 Cricket World Cup?"
    docs = vectorstore.similarity_search(query, k=2)
    retrieved_context = docs[0].page_content
    
    augmented_prompt = f"""

    Given the context below, answer the question. If you don't know the answer, say you don't know.

    Question: {query}

    Context: {retrieved_context}

    Remember to answer only based on the context provided and not from any other source.
    If the question cannot be answered based on the provided context, say I don't know.
    """
    print("\nAugmented Prompt:")
    print(augmented_prompt)

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0, max_tokens=500, timeout=None, max_retries=2)
    response = llm.invoke([("human", augmented_prompt)])
    print("\nResponse:")
    print(response.content)

if __name__ == "__main__":
    main()
