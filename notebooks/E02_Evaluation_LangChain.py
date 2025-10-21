# %% [markdown]
# # Evaluation Part 1: LangChain-Based Approach
#
# ## Purpose
# Test the LangChain-native approach for better RAGAS compatibility
#
# ## Approach
# - Use LangChain's Qdrant.from_documents() directly
# - Use .as_retriever() for retrieval
# - Follow LangChain's retrieval chain pattern

# %%
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.rag.evaluation_langchain import LangChainRAGEvaluator
from app.core.config import settings

print("✓ Imports successful")

# %%
# Initialize evaluator
evaluator = LangChainRAGEvaluator(sgd_path="golden_dataset/sgd_benchmark.csv")

print("✓ Evaluator initialized")

# %% [markdown]
# ## Test 1: Naive Chunking + Naive Retrieval (k=10)

# %%
print("🔄 Evaluating: Naive Chunking + Naive Retrieval (k=10)")
print("=" * 60)

result_naive_10 = evaluator.evaluate_naive_retrieval(
    chunking_strategy="naive",
    chunking_params={
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    },
    top_k=10,
)

print(f"\n✓ Evaluation complete in {result_naive_10['execution_time_seconds']:.2f}s")
print(f"\n📊 RAGAS Metrics (k=10):")
for metric, value in result_naive_10['metrics'].items():
    print(f"  {metric}: {value:.4f}")

# %% [markdown]
# ## Test 2: Semantic Chunking + Naive Retrieval (k=10)

# %%
print("\n🔄 Evaluating: Semantic Chunking + Naive Retrieval (k=10)")
print("=" * 60)

result_semantic_10 = evaluator.evaluate_naive_retrieval(
    chunking_strategy="semantic",
    chunking_params={
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": 95.0,
    },
    top_k=10,
)

print(f"\n✓ Evaluation complete in {result_semantic_10['execution_time_seconds']:.2f}s")
print(f"\n📊 RAGAS Metrics (k=10):")
for metric, value in result_semantic_10['metrics'].items():
    print(f"  {metric}: {value:.4f}")

# %% [markdown]
# ## Test 3: Naive Chunking + BM25 Retrieval (k=10)

# %%
print("\n🔄 Evaluating: Naive Chunking + BM25 Retrieval (k=10)")
print("=" * 60)

result_naive_bm25 = evaluator.evaluate_bm25_retrieval(
    chunking_strategy="naive",
    chunking_params={
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    },
    top_k=10,
)

print(f"\n✓ Evaluation complete in {result_naive_bm25['execution_time_seconds']:.2f}s")
print(f"\n📊 RAGAS Metrics (k=10):")
for metric, value in result_naive_bm25['metrics'].items():
    print(f"  {metric}: {value:.4f}")

# %% [markdown]
# ## Test 4: Semantic Chunking + BM25 Retrieval (k=10)

# %%
print("\n🔄 Evaluating: Semantic Chunking + BM25 Retrieval (k=10)")
print("=" * 60)

result_semantic_bm25 = evaluator.evaluate_bm25_retrieval(
    chunking_strategy="semantic",
    chunking_params={
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": 95.0,
    },
    top_k=10,
)

print(f"\n✓ Evaluation complete in {result_semantic_bm25['execution_time_seconds']:.2f}s")
print(f"\n📊 RAGAS Metrics (k=10):")
for metric, value in result_semantic_bm25['metrics'].items():
    print(f"  {metric}: {value:.4f}")

# %% [markdown]
# ## Full Comparison

# %%
import pandas as pd

comparison_df = pd.DataFrame([
    {
        "Configuration": "Naive Chunking + Vector Retrieval",
        "Chunking": "naive",
        "Retriever": "vector",
        "k": result_naive_10['top_k'],
        **result_naive_10['metrics'],
        "Time (s)": result_naive_10['execution_time_seconds'],
    },
    {
        "Configuration": "Naive Chunking + BM25 Retrieval",
        "Chunking": "naive",
        "Retriever": "bm25",
        "k": result_naive_bm25['top_k'],
        **result_naive_bm25['metrics'],
        "Time (s)": result_naive_bm25['execution_time_seconds'],
    },
    {
        "Configuration": "Semantic Chunking + Vector Retrieval",
        "Chunking": "semantic",
        "Retriever": "vector",
        "k": result_semantic_10['top_k'],
        **result_semantic_10['metrics'],
        "Time (s)": result_semantic_10['execution_time_seconds'],
    },
    {
        "Configuration": "Semantic Chunking + BM25 Retrieval",
        "Chunking": "semantic",
        "Retriever": "bm25",
        "k": result_semantic_bm25['top_k'],
        **result_semantic_bm25['metrics'],
        "Time (s)": result_semantic_bm25['execution_time_seconds'],
    },
])

print("\n📊 Full Comparison Results")
print("=" * 100)
print(comparison_df.to_string(index=False))

# Determine winners
print("\n🏆 Rankings by Metric")
print("=" * 100)

for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
    ranked = comparison_df.sort_values(metric, ascending=False)
    print(f"\n{metric.replace('_', ' ').title()}:")
    for idx, row in ranked.head(3).iterrows():
        print(f"  {idx + 1}. {row['Configuration']}: {row[metric]:.4f}")

# Overall best
best_recall = comparison_df.loc[comparison_df['context_recall'].idxmax()]
best_faithfulness = comparison_df.loc[comparison_df['faithfulness'].idxmax()]

print(f"\n🏆 Best Context Recall: {best_recall['Configuration']}")
print(f"   Recall: {best_recall['context_recall']:.4f}")

print(f"\n🏆 Best Faithfulness: {best_faithfulness['Configuration']}")
print(f"   Faithfulness: {best_faithfulness['faithfulness']:.4f}")

print("\n✓ Complete 4-configuration evaluation finished!")
