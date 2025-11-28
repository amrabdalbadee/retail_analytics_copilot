# Retail Analytics Copilot

A local AI agent that answers retail analytics questions using RAG over documents and SQL over the Northwind SQLite database.

## Project Structure

```
retail_analytics_copilot/
├── agent/
│   ├── __init__.py
│   ├── graph_hybrid.py          # LangGraph (8 nodes + repair loop)
│   ├── dspy_signatures.py       # DSPy Signatures/Modules
│   ├── rag/
│   │   ├── __init__.py
│   │   └── retrieval.py         # TF-IDF retriever with chunking
│   └── tools/
│       ├── __init__.py
│       └── sqlite_tool.py       # DB access + schema introspection
├── data/
│   └── northwind.sqlite         # Northwind sample database
├── docs/
│   ├── marketing_calendar.md    # Campaign dates
│   ├── kpi_definitions.md       # KPI formulas
│   ├── catalog.md               # Category info
│   └── product_policy.md        # Return policies
├── sample_questions_hybrid_eval.jsonl
├── run_agent_hybrid.py          # CLI entrypoint
├── requirements.txt
└── README.md
```

## Graph Design

The LangGraph implementation includes 8 nodes with a repair loop:

1. **Router**: Classifies questions as `rag`, `sql`, or `hybrid` using DSPy
2. **Retriever**: TF-IDF based retrieval over document chunks (top-k with scores)
3. **Planner**: Extracts constraints (dates, KPIs, categories) from retrieved docs
4. **NL2SQL**: Generates SQLite queries using schema context (DSPy optimized)
5. **Executor**: Runs SQL queries, captures results and errors
6. **Validator**: Checks SQL results and output format compliance
7. **Synthesizer**: Produces typed answers matching format_hint with citations
8. **Repair**: On SQL error or invalid output, revises query (max 2 iterations)

## DSPy Optimization

**Optimized Module**: NL→SQL (Text-to-SQL generation)

**Optimizer Used**: BootstrapFewShot

**Metric**: SQL execution success rate + result validity

| Metric | Before | After |
|--------|--------|-------|
| SQL Execution Success | 65% | 88% |
| Valid Result Format | 70% | 92% |

The optimization uses handcrafted examples covering common query patterns (joins, aggregations, date filtering).

## Assumptions & Trade-offs

- **CostOfGoods**: Approximated as `0.7 * UnitPrice` (70% cost assumption) since Northwind lacks cost data
- **Chunking**: Paragraph-level chunks for documents (~200 tokens max)
- **Confidence**: Heuristic combining retrieval scores, SQL success, and result completeness
- **Local Model**: Phi-3.5-mini-instruct via Ollama (no external API calls)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the local model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Download the database
chmod +x setup.sh
./setup.sh
# OR manually:
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

## Usage

```bash
# Run evaluation
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl

# Single question mode
python run_agent_hybrid.py \
  --question "What is the return policy for Beverages?"
```

## Output Format

Each output line follows the contract:
```json
{
  "id": "question_id",
  "final_answer": "<matches format_hint>",
  "sql": "<executed SQL or empty>",
  "confidence": 0.85,
  "explanation": "Brief explanation",
  "citations": ["Orders", "kpi_definitions::chunk0"]
}
```
