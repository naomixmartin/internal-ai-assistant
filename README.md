# Internal AI Assistant

This repo contains an internal AI assistant built with retrieval-augmented generation (RAG), persistent conversation memory, source-grounded citations, and role-based document access.

The system routes queries based on whether company context is required, retrieves authorized information from a vector database, and generates responses grounded in internal company documents. I also built an automated evaluation pipeline to measure retrieval quality, answer correctness, groundedness, citation reliability, permission enforcement, latency, and cost.

For a detailed walkthrough of the system's development, design decisions, scaling challenges, and failure analysis, see [Development Process](docs/development_process.md).

TODO: Add demo GIF or screenshot. Example login → company-specific question → response with citations


## Architecture
TODO: ARCHITECTURE DIAGRAM 
<!-- TODO: Add architecture diagram -->
<!--
Include:
- Streamlit frontend
- Supabase authentication
- LLM router
- embedding generation
- pgvector/HNSW retrieval
- role-based document filtering
- LLM generation
- conversation/message storage
- conversation summarization
- observability
-->

The assistant uses a routed RAG architecture:

1. **Query routing** — Determines whether a request requires company-specific information and rewrites company-related questions into self-contained retrieval queries.
2. **Embedding** — Converts the retrieval query into a 768-dimensional embedding. 
3. Permission-aware retrieval — PostgreSQL and pgvector retrieve semantically similar document chunks using HNSW approximate nearest-neighbor search while filtering documents according to the user's role.
4. **Grounded generation** — Retrieved document chunks are passed to the LLM with numbered sources for citation.
5. **Conversation memory** — Recent messages are retained directly while older conversation history is periodically summarized.
6. **Observability** — Routing, retrieval, and generation are measured independently for latency, token usage, retries, and estimated API cost.

### Tech Stack

- Python
- Streamlit
- Supabase
- PostgreSQL
- pgvector
- Gemini API

---

## Key Features

### Permission-Aware RAG

Company documents are split into overlapping chunks and converted into 768-dimensional embeddings during ingestion. For company-specific queries, the user's question is rewritten into a self-contained retrieval query and embedded using the same embedding model. PostgreSQL/pgvector then uses vector similarity search with an HNSW index to retrieve the most semantically relevant document chunks.

Retrieval is permission-aware: documents are assigned `employee`, `manager`, `hr`, or `admin` access levels, and access restrictions are applied during retrieval so unauthorized documents are never passed to the LLM.

### Query Routing

Using an LLM call, each request is classified as either:

- `GENERAL`
- `COMPANY_CONTEXT_REQUIRED`

For company-specific follow-up questions, conversational context is used to rewrite ambiguous references into self-contained retrieval queries before embedding and vector search.

### Source-Grounded Citations

Retrieved documents are assigned numbered source identifiers. The model cites sources using identifiers such as `[1]` and `[2]`, which are mapped back to the original documents by the application. This prevents the model from generating arbitrary filenames, citing sources with long filenames, and makes citation validity deterministic.

### Persistent Conversation Memory

Conversations and messages are stored in PostgreSQL. Recent messages are included directly in model context, while older messages are periodically summarized to prevent conversation history from growing indefinitely.

### Observability

The pipeline records metrics including:

- routing, embedding, vector search, retrieval, and generation latency
- input and output tokens
- retries and handled API errors
- estimated API cost

---

## Evaluation

I built a 50-case evaluation suite containing company-specific questions, permission-sensitive queries, and general-purpose requests. The evaluation measures the system at multiple stages rather than using final answer accuracy alone. Evaluation metrics include:

| Metric | Purpose |
|---|---|
| Router accuracy | Was the query routed correctly? |
| Recall@K | Was the expected document retrieved within the top K chunks? |
| Groundedness | Were company-specific claims supported by retrieved context? |
| Answer correctness | Did the final response answer the evaluation question? |
| Citation validity | Did cited sources actually come from retrieval? |
| Citation support | Did the cited documents support the associated claims? |
| Permission accuracy | Did retrieval respect the user's document permissions? |

### 5,000-Document Evaluation

The current test corpus contains **5,000 documents and approximately 35,000 document chunks**.

| Metric | Result |
|---|---:|
| Router accuracy | 100% |
| Recall@5 | 82% |
| Recall@10 | 88% |
| Recall@20 | 90% |
| Recall@50 | 94% |
| Answer correctness | 90% |
| Groundedness | 100% |
| Citation validity | 100% |
| Citation presence | 100% |
| Citation support | 100% |
| Permission accuracy | 100% |

<!-- TODO: Optional evaluation summary graphic -->

The 90% answer-correctness score should be interpreted with some caution. Manual review of failed cases identified limitations in the evaluation procedure, including highly specific questions and reference criteria that could produce false negatives. Other failures were caused by the required evidence not reaching the generator rather than the generator incorrectly reasoning over the evidence it received.

For this reason, I treat answer correctness as one part of the evaluation rather than a standalone measure of generation quality. The combination of retrieval recall, answer correctness, groundedness, and citation support gives a more complete picture of system behavior.

See **[Detailed Evaluation & Error Analysis](docs/evaluation.md)** for the evaluation methodology, case-level failure analysis, and limitations of the current evaluation procedure.

---

## Scaling Experiment: 1,000 → 5,000 Documents

I also tested how retrieval quality changed when increasing the corpus from 1,000 to 5,000 documents while keeping the evaluation set fixed.

<!-- TODO: Add 1K vs. 5K Recall@K plot -->
<!--
RECOMMENDED FIGURE

x-axis: K = 5, 10, 20, 50
y-axis: Recall (%)

1K: 94, 94, 94, 94
5K: 82, 88, 90, 94
-->

| Metric | 1K Documents | 5K Documents | Change |
|---|---:|---:|---:|
| Router accuracy | 100% | 100% | — |
| Recall@5 | 94% | 82% | -12 pp |
| Recall@10 | 94% | 88% | -6 pp |
| Recall@20 | 94% | 90% | -4 pp |
| Recall@50 | 94% | 94% | — |
| Answer correctness | 94% | 90% | -4 pp |
| Groundedness | 100% | 100% | — |
| Citation support | 100% | 100% | — |
| Permission accuracy | 100% | 100% | — |

The largest degradation occurred at shallow retrieval depths. Recall@5 fell from 94% to 82%, while Recall@50 remained at 94%.

This suggests that scaling primarily created a **ranking problem rather than a candidate-availability problem**. Relevant documents were generally still present in the larger candidate set, but additional documents pushed some of them lower in the ranking.

Despite increasing the corpus size by 5×, answer correctness decreased by only four percentage points, while groundedness, citation support, and permission accuracy remained unchanged.

More detailed retrieval and failure analysis is included in **[docs/evaluation.md](docs/evaluation.md)**.

---


## Error Analysis

Manual review of failed evaluation cases showed that not all end-to-end failures came from the same component.

Two important retrieval failure modes were:

**Ranking failures** — the required evidence was retrieved but ranked below the top-10 chunks supplied to the generator.

**Candidate retrieval failures** — the answer-bearing evidence was absent even from a larger candidate set.

The distinction matters because the solutions are different. Reranking could potentially improve the first case, while evidence missing from the candidate set requires changes earlier in the retrieval pipeline.

The analysis also identified limitations in the evaluation procedure itself. Some responses marked incorrect were reasonable and grounded given the retrieved evidence but failed against overly specific evaluation criteria.

Rather than reproducing the full case-by-case analysis here, I document these failures and evaluation limitations in **[Detailed Evaluation & Error Analysis](docs/evaluation.md)**.

<!-- TODO: Optional failure-mode diagram -->
<!--
                    Failed evaluation
                          |
            +-------------+-------------+
            |             |             |
         Ranking       Candidate     Evaluation
         failure       retrieval      artifact
                          failure
-->

---

## Performance and Cost

Latency is measured independently across routing, embedding, vector search, retrieval, and generation to identify pipeline bottlenecks.

| Stage | p50 Latency | p95 Latency |
| --- | ---: | ---: |
| Routing | TBD | TBD |
| Embedding | TBD | TBD |
| Vector search | TBD | TBD |
| Retrieval | TBD | TBD |
| Generation | TBD | TBD |
| Total | TBD | TBD |

API token usage, retries, handled errors, and estimated cost are also recorded for each request.

TODO FIGURE: Horizontal bar chart of median latency by pipeline stage.

> **Performance experiments are ongoing.** Final latency and cost results will be added after completing the current model and pipeline optimization experiments.

---

## Future Work

- compare LLM and non-LLM routing approaches
- compare LLM models on quality, latency, and cost
- evaluate reranking for top-K retrieval
- improve frontend responsiveness
- add project- or team-level document permissions

## Running Locally

TODO: Add installation and environment setup instructions
*Setup instructions forthcoming.*

## Project Status

Active development. Core RAG, permissions, persistent memory, citations, automated evaluation, corpus-scaling experiments, and observability are implemented. Performance optimization and additional evaluation work are ongoing.