# Architecture Decision Record: Orchestration Strategy

**Status:** Accepted  
**Date:** 2025-10-22  
**Decision Makers:** Architecture Team  
**Related Documents:** 
- [Certification Challenge Report](../../certification_challenge_report.md)
- [Architecture Document](../archtecture.md)
- [Orchestration Implementation](../../backend/app/orchestration/pipeline.py)

---

## Context

Legal-OS implements a multi-agent system for M&A due diligence with 5 specialized agents (Clause Extraction, Risk Scoring, Summary, Checklist, and Source Tracker). We needed to make two critical architectural decisions:

1. **Agent Execution Pattern**: Sequential vs. Parallel orchestration
2. **Graph Implementation**: When to use LangGraph's `create_react_agent` vs. custom graph implementations

These decisions significantly impact system complexity, performance, maintainability, and debugging capabilities.

---

## Decision 1: Sequential Agent Orchestration

### Decision

**We use sequential orchestration where the supervisor agent calls specialized agents one at a time in a defined order.**

**Execution Flow:**
```
Ingestion → Clause Extraction → Risk Scoring → Summary → Checklist
```

### Rationale

#### 1. **Data Dependencies**
Each agent depends on the previous agent's output:
- **Risk Scoring** requires extracted clauses from **Clause Extraction**
- **Summary** requires risk scores from **Risk Scoring**
- **Checklist** requires the complete analysis from **Summary**

Parallel execution would require complex synchronization and duplicate work.

#### 2. **State Management Simplicity**
Sequential execution provides:
- **Clear state transitions**: Each agent receives well-defined input from previous agent
- **Easier debugging**: Linear execution path makes it simple to identify where failures occur
- **Deterministic results**: Same input always produces same output in same order
- **Simplified checkpointing**: Can save state after each agent completes

#### 3. **Cost Optimization**
- **Avoids redundant LLM calls**: Parallel agents might query the same information multiple times
- **Efficient token usage**: Each agent builds on previous results rather than re-analyzing
- **Predictable costs**: Sequential execution makes cost estimation straightforward

#### 4. **Domain Alignment**
M&A due diligence is inherently sequential:
- You can't score risks before identifying clauses
- You can't write a summary before understanding risks
- You can't create a checklist before completing analysis

The sequential pattern mirrors the actual legal workflow.

#### 5. **Error Handling**
- **Precise failure identification**: Know exactly which agent failed
- **Granular retry logic**: Can retry from failed agent without re-running entire pipeline
- **Partial results**: Can return results from completed agents even if later agents fail
- **Easier testing**: Can test each agent independently with mock inputs

#### 6. **Observability**
- **Clear progress tracking**: Can report "3 of 5 agents complete"
- **Better logging**: Linear execution produces readable, chronological logs
- **LangSmith tracing**: Sequential traces are easier to understand and debug

### Alternatives Considered

#### Alternative 1: Parallel Execution
**Approach**: Run all agents simultaneously, merge results at end

**Pros**:
- Faster execution (agents run concurrently)
- Better resource utilization

**Cons**:
- Complex state synchronization
- Duplicate LLM calls (higher cost)
- Difficult to debug race conditions
- Non-deterministic results
- Doesn't match domain workflow

**Decision**: Rejected due to complexity and cost

#### Alternative 2: Hybrid (Parallel + Sequential)
**Approach**: Parallel where possible, sequential where required

**Pros**:
- Balance of speed and simplicity
- Optimize independent tasks

**Cons**:
- Increased complexity
- Harder to reason about execution flow
- Premature optimization for MVP

**Decision**: Deferred to future optimization (when we have multi-document analysis)

### Consequences

#### Positive
- ✅ Simple, maintainable codebase
- ✅ Predictable execution and costs
- ✅ Easy debugging and testing
- ✅ Clear progress reporting
- ✅ Matches domain workflow

#### Negative
- ⚠️ Slower than parallel execution (acceptable for MVP)
- ⚠️ Single-threaded bottleneck (can optimize later)

#### Neutral
- 🔄 Can parallelize within agents (e.g., multiple document chunks)
- 🔄 Can add parallel execution for independent tasks in future

### Implementation

```python
# backend/app/orchestration/pipeline.py
class DocumentOrchestrator:
    def run_orchestration(self, document_path: str, document_id: str):
        # Sequential execution
        results = {}
        
        # Step 1: Ingestion
        ingestion_result = self.ingestion_agent.process(document_path)
        results['ingestion'] = ingestion_result
        
        # Step 2: Clause Extraction (depends on ingestion)
        clause_result = self.clause_agent.extract_clauses(
            document_id=ingestion_result['document_id']
        )
        results['clauses'] = clause_result
        
        # Step 3: Risk Scoring (depends on clauses)
        risk_result = self.risk_agent.score_risks(
            clause_extraction_result=clause_result
        )
        results['risks'] = risk_result
        
        # Step 4: Summary (depends on risks)
        summary_result = self.summary_agent.generate_summary(
            risk_scoring_result=risk_result
        )
        results['summary'] = summary_result
        
        # Step 5: Checklist (depends on summary)
        checklist_result = self.checklist_agent.generate_checklist(
            summary=summary_result
        )
        results['checklist'] = checklist_result
        
        return results
```

---

## Decision 2: Mixed Graph Implementation Strategy

### Decision

**We use a mixed approach:**
- **`create_react_agent`** for simple, linear workflows (Risk Scoring, Summary, Checklist)
- **Custom LangGraph** for complex, multi-step reasoning (Clause Extraction)

### Rationale

#### When to Use `create_react_agent`

**Use for agents with:**
- Simple, linear tool use patterns
- Standard ReAct loop (Thought → Action → Observation)
- Minimal custom logic between steps
- Standard error handling sufficient

**Benefits:**
- **Less boilerplate**: LangGraph handles ReAct loop automatically
- **Built-in features**: Automatic retries, error handling, state management
- **Faster development**: Focus on tools, not graph structure
- **Standard patterns**: Easier for other developers to understand
- **Better defaults**: Optimized ReAct implementation

**Example Agents:**
- **Risk Scoring Agent**: Simple rule application with LLM reasoning
- **Summary Agent**: Straightforward synthesis with optional retrieval
- **Checklist Agent**: Linear analysis → checklist generation

#### When to Build Custom Graphs

**Use for agents with:**
- Complex, multi-step workflows
- Custom state transitions
- Domain-specific reasoning patterns
- Fine-grained control requirements
- Custom error handling or validation

**Benefits:**
- **Full control**: Define exact state transitions
- **Custom logic**: Implement domain-specific reasoning
- **Better visualization**: Graph structure shows complex workflow
- **Optimized performance**: Eliminate unnecessary steps
- **Easier debugging**: See exactly what's happening at each step

**Example: Clause Extraction Agent**

Why it needs a custom graph:
1. **Multi-query strategy**: Tries different search queries if first fails
2. **Validation loops**: Checks extracted clauses against schema, retries if invalid
3. **Deduplication**: Merges similar clauses found via different queries
4. **Partial success handling**: Returns partial results if some clause types fail
5. **Confidence scoring**: Custom logic to assess extraction quality

### Implementation Examples

#### Simple Agent with `create_react_agent`

```python
# backend/app/agents/risk_scoring.py
from langgraph.prebuilt import create_react_agent

class RiskScoringAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.tools = [self._create_risk_scoring_tool()]
        
        # Use create_react_agent for simple workflow
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            state_modifier="You are a legal risk assessment expert..."
        )
    
    def score_risks(self, clause_extraction_result):
        # Simple invocation - ReAct loop handled automatically
        result = self.agent.invoke({
            "messages": [HumanMessage(content=f"Score risks for: {clause_extraction_result}")]
        })
        return self._parse_result(result)
```

#### Complex Agent with Custom Graph

```python
# backend/app/agents/clause_extraction.py
from langgraph.graph import StateGraph, END

class ClauseExtractionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Build custom graph for complex workflow
        workflow = StateGraph(ClauseExtractionState)
        
        # Define custom nodes
        workflow.add_node("initial_search", self._initial_search)
        workflow.add_node("validate_clauses", self._validate_clauses)
        workflow.add_node("retry_search", self._retry_with_different_query)
        workflow.add_node("deduplicate", self._deduplicate_clauses)
        workflow.add_node("score_confidence", self._score_confidence)
        
        # Define custom edges with conditional logic
        workflow.set_entry_point("initial_search")
        workflow.add_conditional_edges(
            "validate_clauses",
            self._should_retry,
            {
                "retry": "retry_search",
                "continue": "deduplicate"
            }
        )
        workflow.add_edge("deduplicate", "score_confidence")
        workflow.add_edge("score_confidence", END)
        
        self.graph = workflow.compile()
    
    def _should_retry(self, state):
        """Custom logic: retry if validation fails and attempts < 3"""
        if not state["validation_passed"] and state["attempts"] < 3:
            return "retry"
        return "continue"
```

### Decision Matrix

| Criteria | create_react_agent | Custom Graph |
|----------|-------------------|--------------|
| **Workflow Complexity** | Simple, linear | Complex, branching |
| **Custom Logic** | Minimal | Extensive |
| **State Transitions** | Standard | Custom |
| **Error Handling** | Built-in sufficient | Custom required |
| **Development Speed** | Fast | Slower |
| **Debugging** | Standard tools | Custom visualization |
| **Maintenance** | Easy | Moderate |
| **Performance** | Good | Optimized |

### Consequences

#### Positive
- ✅ **Best of both worlds**: Simplicity where possible, control where needed
- ✅ **Faster development**: Don't reinvent wheel for simple agents
- ✅ **Better performance**: Optimize complex workflows with custom graphs
- ✅ **Easier maintenance**: Standard patterns for simple agents
- ✅ **Clear guidelines**: Decision matrix helps choose approach

#### Negative
- ⚠️ **Mixed patterns**: Developers need to understand both approaches
- ⚠️ **Consistency**: Different agents use different patterns

#### Neutral
- 🔄 **Flexibility**: Can migrate simple agents to custom graphs if needed
- 🔄 **Evolution**: Start with `create_react_agent`, upgrade to custom if complexity grows

---

## Monitoring and Review

### Success Metrics
- **Execution time**: Sequential pipeline completes in <2 minutes for typical document
- **Error rate**: <5% agent failures in production
- **Debugging time**: Issues identified and fixed within 1 hour
- **Development velocity**: New agents added in <2 days

### Review Triggers
- If execution time exceeds 5 minutes → Consider parallelization
- If error rate exceeds 10% → Review error handling strategy
- If debugging takes >4 hours → Improve observability
- If new agent takes >5 days → Review graph complexity

### Future Considerations

#### Potential Optimizations
1. **Parallel execution for independent tasks**
   - Multi-document analysis (compare multiple contracts)
   - Batch processing (analyze 10 contracts simultaneously)
   
2. **Streaming results**
   - Return partial results as agents complete
   - Update UI in real-time
   
3. **Caching**
   - Cache clause extraction results for similar documents
   - Reuse risk scoring for identical clauses

4. **Adaptive orchestration**
   - Skip agents based on document type
   - Parallel execution for independent analysis paths

---

## References

### Code
- [Orchestration Implementation](../../backend/app/orchestration/pipeline.py)
- [Clause Extraction (Custom Graph)](../../backend/app/agents/clause_extraction.py)
- [Risk Scoring (create_react_agent)](../../backend/app/agents/risk_scoring.py)
- [Summary Agent (create_react_agent)](../../backend/app/agents/summary.py)
- [Checklist Agent (create_react_agent)](../../backend/app/agents/checklist.py)

### Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Pattern](https://arxiv.org/abs/2210.03629)
- [Architecture Document](../archtecture.md)

### Related Decisions
- [Chunking Strategy Selection](./chunking-strategy-selection.md)
- [Base Retriever Selection](./base-retriever-selection.md)
- [Final Evaluation Results](./final-evaluation-results.md)

---

## Appendix: Performance Comparison

### Sequential vs. Parallel (Estimated)

| Metric | Sequential | Parallel | Notes |
|--------|-----------|----------|-------|
| **Execution Time** | 120s | 45s | Parallel 2.7x faster |
| **LLM Calls** | 15 | 25 | Parallel 67% more calls |
| **Cost per Run** | $0.08 | $0.13 | Parallel 63% more expensive |
| **Debugging Time** | 15min | 60min | Sequential 4x easier |
| **Code Complexity** | Low | High | Parallel needs sync logic |

**Conclusion**: For MVP, sequential execution provides better cost-efficiency and maintainability. Parallel execution can be added later for performance-critical use cases.

---

