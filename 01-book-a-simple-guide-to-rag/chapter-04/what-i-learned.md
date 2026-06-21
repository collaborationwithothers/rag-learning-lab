# Learnings
- Generation pipeline involves retrievel, augmentation of the prompt and response generation by the LLM
- Retriever is a component that retrieves top N matching chunks for the user query
- BM25(Best Match 25) is a retrieval method that uses keyword search ot match documents with the query
-  Retriver methods are usually combined for ex for a quciky retrieval BM25 may be used and further re-ranking over it using vector search