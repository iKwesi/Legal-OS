# Chunking Strategy Selection Decision

## Decision Date
2025-10-20

## Context
Story 2.1 required evaluation of different chunking strategies to determine the optimal approach for the Legal-OS RAG pipeline.

## Options Evaluated

### 1. Naive Chunking (RecursiveCharacterTextSplitter)
- **Implementation**: LangChain's RecursiveCharacterTextSplitter
- **Parameters**: chunk_size=1000, chunk_overlap=200
- **Pros**: Fast, predictable, simple
- **Cons**: May split semantic units inappropriately

### 2. Semantic Chunking
- **Implementation**: Sentence-based semantic boundary detection
- **Parameters**: breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95.0
- **Pros**: Preserves semantic coherence, better context boundaries
- **Cons**: More computationally expensive

## Evaluation Results

Tested with Vector Similarity retrieval (k=10):

| Chunking Strategy | Precision | Recall | Faithfulness | Relevancy |
|-------------------|-----------|--------|--------------|-----------|
| **Semantic** 🏆 | 87.47% | **66.67%** | **92.53%** | 87.30% |
| Naive | **93.04%** | 63.33% | 87.62% | **87.59%** |

## Decision

**Selected: Semantic Chunking**

### Configuration
```python
{
    "strategy": "semantic",
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": 95.0
}
```

### Justification

1. **Higher Context Recall** (66.67% vs 63.33%)
   - Retrieves 5% more relevant information
   - Critical for comprehensive legal analysis

2. **Higher Faithfulness** (92.53% vs 87.62%)
   - 6% improvement in answer accuracy
   - Answers more grounded in actual document content

3. **Better Semantic Coherence**
   - Preserves meaning across chunk boundaries
   - Maintains legal context integrity
   - Reduces fragmented information

4. **Acceptable Trade-offs**
   - Precision: -5.57% (87.47% vs 93.04%) - still excellent
   - Relevancy: -0.29% (87.30% vs 87.59%) - negligible difference

### When to Use

- **Default**: Semantic chunking for all legal document processing
- **Alternative**: Naive chunking only if performance is critical and slight quality reduction is acceptable

## Implementation

```python
from app.rag.chunking import get_chunker

chunker = get_chunker(
    strategy="semantic",
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0
)
```

## References

- Full evaluation results: [final-evaluation-results.md](./final-evaluation-results.md)
- Evaluation notebook: `notebooks/E02_Evaluation_LangChain.py`
- Story: [2.1.evaluate-chunking-strategies-select-base-retriever.md](../stories/2.1.evaluate-chunking-strategies-select-base-retriever.md)
