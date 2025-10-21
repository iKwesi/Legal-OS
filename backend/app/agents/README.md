## Clause Extraction Agent

The Clause Extraction Agent is a specialized LangGraph-based agent that analyzes M&A documents to extract key clauses and detect potential red flags.

### Features

- **ReAct Pattern**: Uses Reasoning and Acting pattern for systematic document analysis
- **LangGraph Orchestration**: State-based workflow with clear reasoning, action, and observation steps
- **Vector Similarity Retrieval**: Integrates with the optimal retriever selected in Story 2.1
- **Structured Output**: Returns typed Pydantic models for easy integration
- **Red Flag Detection**: Identifies potential issues with severity levels (Critical, High, Medium, Low)
- **Visualization Support**: Can generate LangGraph visualizations for debugging

### Usage

```python
from app.agents.clause_extraction import ClauseExtractionAgent
from app.rag.vector_store import VectorStore

# Initialize agent
vector_store = VectorStore()
agent = ClauseExtractionAgent(
    vector_store=vector_store,
    model_name="gpt-4o-mini",
    temperature=0.0,
    top_k=10
)

# Extract clauses from a document
result = agent.extract_clauses(
    document_id="doc_123",
    session_id="session_456"  # Optional
)

# Access results
for clause in result.clauses:
    print(f"{clause.clause_type}: {clause.clause_text}")
    print(f"Confidence: {clause.confidence}")

for red_flag in result.red_flags:
    print(f"{red_flag.severity}: {red_flag.description}")
    print(f"Recommendation: {red_flag.recommendation}")
```

### Clause Types

The agent extracts the following M&A clause types:

1. **payment_terms** - Purchase price, payment schedule, escrow
2. **warranties** - Representations and warranties
3. **indemnification** - Indemnification obligations and caps
4. **termination** - Termination rights and conditions
5. **confidentiality** - Confidentiality obligations
6. **non_compete** - Non-compete restrictions
7. **dispute_resolution** - Governing law and dispute resolution

### Red Flag Severity Levels

- **Critical**: Missing material warranties, unlimited indemnification, no liability cap
- **High**: Vague payment terms, short survival periods, weak protections
- **Medium**: Missing standard representations, ambiguous conditions
- **Low**: Minor inconsistencies, clarification needed

### Visualization

To visualize the agent's LangGraph workflow:

```python
# In a Jupyter notebook
from IPython.display import display

viz = agent.get_graph_visualization()
display(viz)
```

### Architecture

The agent uses a state-based workflow:

1. **Reason Node**: Decides what action to take next based on current state
2. **Act Node**: Executes tools (search, extract, detect)
3. **Observe Node**: Processes tool results and updates state
4. **Finalize Node**: Prepares final results

The agent stops after:
- Maximum iterations (10) reached
- No more tool calls to execute
- All clause types have been analyzed

### Tools

The agent has access to three tools:

1. **search_document**: Semantic search for relevant document chunks
2. **extract_clause**: Extract structured clause information from text
3. **detect_red_flags**: Analyze clauses for potential issues

### Testing

Run the test suite:

```bash
cd backend
uv run pytest tests/test_clause_extraction.py -v
```

### Data Models

See `app/models/agent.py` for complete data model definitions:

- `ExtractedClause`: Represents a single extracted clause
- `RedFlag`: Represents a detected issue
- `ClauseExtractionResult`: Complete agent output
