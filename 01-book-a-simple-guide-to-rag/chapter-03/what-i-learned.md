# Version migration notes

The book used LangChain 0.3.19.

I am using latest LangChain.

Changes I made:
- I did not install packages inside the notebook.
- I used uv to manage dependencies.
- I used `langchain_community.document_loaders` for `AsyncHtmlLoader`.
- I kept the notebook for experimentation and `main.py` for clean runnable code.

# Learnings
- Data ingestion involves getting data from various disparate sources and in different formats
- Some documents may have metadata and some may not have
- Data ingestion also involves cleanup
- Deduplication
- Removal of sensitive and PII data

## Chunking
- Chunking is the next stage in data ingestion. The source data might be quite large and it needs to be broken down into manageable chunks, but manageable for who?
    - LLM context window - LLMs have context window sizes. Not every size of data can fit into context
    - Retrieval needs to be efficient and quick and this relates to the sizes of chunks
    - Context accuracy also depends on the size of the chunk

## Chunking considerations
- Need to consider on what basis to split the content
- Need to determine the size of the chunk
- Need to determine the overlap between the chunks to maintain context
- If chucking size is too small, then context is lost and also might introduce latency as multiple retrievals may be needed
- If chunks are too large, that will have an impact on context window

## Chunking methods
- Fixed size
    - The chunk size and splitting are predetermined
- Specialized chunking
    - Chunking is determined based in data format and structure
    - For example, langchain offers Htlm deaders based splitter, json based splitter etc
- Semantic chunking
    - All the other chunking methods do not consider the semantic meaning of the content while chunking. Smentic chunking considers the semantic meaning so that it preserves the meaning

## Factors affecting chunking strategy
- Format of the data and the data source
- Size of the prompts
- Nature of the application use case
    - Direct question and answer may benefit from short chunks
    - Summarization tasks may benefit from longer chunks
- Embedding models used as some models affect the size of chunks
- Always evaluate data ingestion pipeline

## Embeddings
- Embeddings are multi dimensiona vector representations of the chunks
- There are specilised embedding models
- Choice of mebedding model also determins retrieval quality
- There are different models and they serve different use cases
- Similarity between the user prompt and knowledge is determined using cosine or eucledean similarity
