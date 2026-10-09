# Internal AI Assistant

This repository contains an internal AI assistant built with retrieval-augmented generation (RAG), persistent conversation memory, source-grounded citations, and role-based document access.

The system routes queries based on whether company context is required, retrieves authorized information from a vector database, and generates responses grounded in internal company documents. I also built an automated evaluation pipeline to measure retrieval quality, answer correctness, groundedness, citation reliability, permission enforcement, latency, and cost.

A primary motivation for this project was exploring how an internal AI assistant could maintain context beyond an individual user's conversations. In collaborative environments, useful knowledge is distributed across employees, teams, projects, and documents. Shared organizational memory could allow this context to persist and become available to other authorized users rather than remaining isolated within individual conversations.

The current architecture provides a foundation for this capability. The same retrieval and authorization mechanisms could be extended to organization-wide, team, and project-level memory, with explicit access and privacy boundaries controlling which context can be shared.

For a detailed walkthrough of the system's development, design decisions, scaling challenges, and failure analysis, see [Development Process](docs/development_process.md).

![Internal AI Assistant demo](results/demo.gif)

## Tech Stack

- Python
- Streamlit
- Supabase (PostgreSQL, Auth, RLS)
- pgvector (HNSW indexing)
- Gemini API

---

## Architecture
The assistant uses a routed RAG architecture:

1. **Query routing** — Determines whether a request requires company-specific information and rewrites company-related questions into self-contained retrieval queries.
2. **Embedding** — Converts the retrieval query into a 768-dimensional embedding. 
3. **Permission-aware retrieval** — PostgreSQL and pgvector retrieve semantically similar document chunks using HNSW approximate nearest-neighbor search while filtering documents according to the user's role.
4. **Grounded generation** — Retrieved document chunks are passed to the LLM with numbered sources for citation.
5. **Conversation memory** — Recent messages are retained directly while older conversation history is periodically summarized.
6. **Observability** — Routing, retrieval, and generation are measured independently for latency, token usage, retries, and estimated API cost.

The full architecture is as follows: 


```text
DOCUMENT INGESTION
──────────────────

Company Documents
        ↓
 Text Extraction
        ↓
 Document Chunking
(250 words, 50 overlap)
        ↓
Gemini Embedding 2
(768D embeddings)
        ↓
┌──────────────────────────────┐
│Supabase / PostgreSQL Database│
│                              │
│    Documents + Chunks        │
│    pgvector + HNSW           │
│    Role-Based Permissions    │
└──────────────┬───────────────┘
               │
               │
               ↓
    ONLINE QUERY PIPELINE
    ─────────────────────

         User Message
               ↓
       Conversation Memory
   (recent messages + summary)
               ↓
           LLM Router
               │
    ┌──────────┴──────────┐
    │                     │
 GENERAL        COMPANY_CONTEXT_REQUIRED
    │                     │
    │                     ↓
    │            Context-Aware Query
    │                  Rewrite
    │                     ↓
    │             Gemini Embedding 2
    │            (768-D Query Vector)
    │                     ↓
    │          Permission-Aware Vector
    │                   Search
    │                     ↓
    │               Top-K Similar Chunks
    │                     ↓
    │             Conversation Context
    │             + Retrieved Chunks
    │                     │
    └──────────┬──────────┘
               ↓
          Answering LLM
               ↓
        Grounded Response
       + Source Citations
```
---

## Key Features

### Permissions-Aware RAG

Company documents are split into overlapping chunks and converted into 768-dimensional embeddings during ingestion. For company-specific queries, the user's question is rewritten into a self-contained retrieval query and embedded using the same embedding model. PostgreSQL/pgvector then uses vector similarity search with an HNSW index to retrieve the most semantically relevant document chunks.

Retrieval is permission-aware: documents are assigned `employee`, `manager`, `hr`, or `admin` access levels, and access restrictions are applied during retrieval so unauthorized documents are never passed to the LLM.

### Query Routing

Using an LLM call, each request is classified as either `GENERAL` or `COMPANY_CONTEXT_REQUIRED`. For company-specific follow-up questions, conversational context is used to rewrite ambiguous references into self-contained retrieval queries before embedding and vector search.

### Source-Grounded Citations

Retrieved documents are assigned numbered source identifiers. The model cites sources using identifiers such as `[1]` and `[2]`, which are mapped back to the original documents by the application. This prevents the model from generating arbitrary filenames, citing sources with long filenames, and simplifies citation validation.

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

| Metric | Purpose                                                                        |
|---|--------------------------------------------------------------------------------|
| Router accuracy | Was the query routed correctly?                                                |
| Retrieval accuracy | Was the expected document retrieved within the production Top-K retrieval set? |
| Recall@K | Was the expected document retrieved within the top K chunks? |
| Groundedness | Were company-specific claims supported by retrieved context?                   |
| Answer correctness | Did the final response answer the evaluation question?                         |
| Citation validity | Did cited sources actually come from retrieval?                                |
| Citation support | Did the cited documents support the associated claims?                         |
| Permission accuracy | Did retrieval respect the user's document permissions?                         |
| Median response latency | What was the median elapsed time between user query submission and response?   |

### 5,000-Document Evaluation

The current test corpus contains 5,000 documents and approximately 35,000 document chunks. Evaluation results are as follows.

| Metric                  | Result |
|-------------------------|-------:|
| Router accuracy         |   100% |
| Retrieval accuracy      |    88% | 
| Recall@10 |    88% |
| Recall@20 |    90% |
| Recall@50 |    94% |
| Answer correctness      |    90% |
| Groundedness            |   100% |
| Citation validity       |   100% |
| Citation support        |   100% |
| Permission accuracy     |   100% |
| Median response latency |  6.25s |

The 90% answer-correctness score should be interpreted with some caution. Manual review of failed cases identified limitations in the evaluation procedure, including highly specific questions and reference criteria that could produce false negatives. Other failures were caused by the required evidence not reaching the generator rather than the generator incorrectly reasoning over the evidence it received.  For this reason, I treat answer correctness as only one part of the evaluation. The combination of retrieval recall, answer correctness, groundedness, and citation support gives a more complete picture of system behavior.

See **[Development Process](docs/development_process.md)** Sections 7 and 12 for complete evaluation methodology, case-level failure analysis, and limitations of the current evaluation procedure.

---

## Error Analysis

Manual review showed that failed evaluation cases fell into several distinct categories:

- **Document-level ranking failures** — the intended document was not retrieved within the top-10 results but appeared at larger retrieval depths.
- **Chunk-level ranking failures** — the correct document was retrieved, but the answer-bearing chunk ranked too low to reach the generator.
- **Candidate retrieval failures** — the required evidence was absent even from the larger retrieved candidate set.
- **Generation failures** — the correct evidence reached the LLM, but competing context caused the generator to select the wrong information.
- **Evaluation ambiguity** — some broad questions had multiple valid answers, while the evaluator expected one specific source or answer.

These distinctions are important because they point to different improvements. The full case-level analysis and potential improvements are documented in **[Development Process](docs/development_process.md)**, Section 12.

---

## Future Work

The current system provides a measured baseline for future improvements: 

- investigate opportunities to reduce overall system latency, particularly routing and generation overhead 
- test reranking or hybrid retrieval against identified retrieval failures
- explore organization-wide and project-level memory to improve context sharing across collaborative work
- compare generation models on answer quality, latency, and cost
- improve frontend responsiveness or replace Streamlit if application latency becomes a priority

---

## Project Status

The core system is complete as an evaluated prototype, with further work ongoing. It implements permission-aware
RAG, persistent conversation memory, contextual query rewriting, grounded citations,
automated evaluation, scalable vector retrieval, and stage-level observability.

The system has been evaluated at up to 5,000 documents, with remaining work focused on 
targeted experiments around retrieval quality, latency, shared 
organizational memory, and production-level application infrastructure.