# Base Retriever Selection Decision

## Decision Date
2025-10-20

## Context
Story 2.1 required evaluation of different retriever types to determine the optimal approach for the Legal-OS RAG pipeline.

## Options Evaluated

### 1. Vector Similarity (Naive Retrieval)
- **Implementation**: LangChain's Qdrant vector similarity search
- **Parameters**: k=10, cosine similarity
- **Pros**: Semantic understanding, handles paraphrasing well
- **Cons**: Requires embeddings (computational cost)

### 2. BM25 (Keyword-Based Retrieval)
- **Implementation**: LangChain's BM25Retriever
- **Parameters**: k=10, k1=1.5, b=0.75
- **Pros**: Excellent for exact term matching, no embeddings needed
- **Cons**: Doesn't understand semantic similarity

## Evaluation Results

Tested with both Naive and Semantic chunking strategies:

### Vector Similarity Results

| Chunking | Precision | Recall | Faithfulness | Relevancy |
|----------|-----------|--------|--------------|-----------|
| Naive | 93.04% | 63.33% | 87.62% | 87.59% |
| Semantic | 87.47% | **66.67%** | **92.53%** | 87.30% |
| **Average** | **90.26%** | **65.00%** | **90.08%** | **87.45%** |

### BM25 Results

| Chunking | Precision | Recall | Faithfulness | Relevancy |
|----------|-----------|--------|--------------|-----------|
| Naive | 74.52% | 50.67% | 70.69% | 57.42% |
| Semantic | 74.43% | 53.17% | 79.30% | 57.50% |
| **Average** | **74.48%** | **51.92%** | **75.00%** | **57.46%** |

## Performance Comparison

| Metric | Vector | BM25 | Improvement |
|--------|--------|------|-------------|
| Context Precision | 90.26% | 74.48% | **+21%** |
| Context Recall | 65.00% | 51.92% | **+25%** |
| Faithfulness | 90.08% | 75.00% | **+20%** |
| Answer Relevancy | 87.45% | 57.46% | **+52%** |

## Decision

**Selected: Vector Similarity Retrieval**

(Also called "Naive Retrieval" in code - refers to embedding-based vector similarity search)

### Configuration
```python
{
    "retriever_type": "vector",
    "search_kwargs": {"k": 10},
    "similarity_metric": "cosine"
}
```

### Justification

1. **Superior Performance Across All Metrics**
   - 21% better precision than BM25
   - 25% better recall than BM25
   - 20% better faithfulness than BM25
   - 52% better answer relevancy than BM25

2. **Semantic Understanding**
   - Handles paraphrased questions effectively
   - Understands legal terminology in context
   - Better for complex legal queries

3. **Production-Quality Metrics**
   - 90% faithfulness - answers grounded in documents
   - 65% context recall - retrieves relevant information
   - 87% answer relevancy - addresses user questions

4. **Consistent Performance**
   - Works well with both chunking strategies
   - Reliable across different query types
   - Predictable behavior

### Why Not BM25?

While BM25 has advantages for exact keyword matching:
- **Lower recall** (52% vs 65%) - misses relevant information
- **Lower faithfulness** (75% vs 90%) - less accurate answers
- **Much lower relevancy** (57% vs 87%) - answers less helpful
- **Not suitable as primary retriever** for legal Q&A

**Note**: BM25 may still be valuable in a **hybrid retrieval** approach (Epic 3) to complement vector search for exact term matching.

## Implementation

### Using LangChain Native Approach

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

# Create vectorstore
vectorstore = Qdrant.from_documents(
    documents,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    location=":memory:",  # or URL for production
    collection_name="legal_documents"
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

### RAG Chain Pattern

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | llm, "context": itemgetter("context")}
)

# Use it
result = rag_chain.invoke({"question": "What are the key terms?"})
```

## Future Considerations

### Epic 3: Advanced Retrieval Strategies

1. **Hybrid Retrieval** (Vector + BM25)
   - Combine semantic and keyword-based search
   - May improve recall further
   - Evaluate in Epic 3

2. **Reranking with Cohere**
   - Post-retrieval reranking
   - May improve precision
   - Evaluate in Epic 3

3. **Query Expansion**
   - Expand queries with legal terminology
   - May improve recall for complex queries
   - Evaluate in Epic 3

## References

- Full evaluation results: [final-evaluation-results.md](./final-evaluation-results.md)
- Evaluation notebook: `notebooks/E02_Evaluation_LangChain.py`
- Story: [2.1.evaluate-chunking-strategies-select-base-retriever.md](../stories/2.1.evaluate-chunking-strategies-select-base-retriever.md)
- Implementation: `backend/app/rag/evaluation_langchain.py`
