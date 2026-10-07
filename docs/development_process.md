# Development Process

This document describes the process of developing my Internal
AI Assistant. The assistant started as a small Streamlit chatbot with basic
LLM functionality and evolved into a permissions-aware, evaluated RAG system
with a 5,000-document corpus. The goal was not to design the final
architecture up front. I built a minimal version first, measured where
it failed, and added complexity only when the system demonstrated a
concrete need for it.

## 1. Starting with the application foundation

I started with a simple goal: build an internal company assistant that
could eventually answer questions using private company information
while maintaining user-specific conversations and access controls.


For the initial prototype, I chose Streamlit because it allowed me to
build a functional chat interface entirely in Python and focus development 
time on the backend and ML system rather than frontend engineering. 
I chose **Supabase/PostgreSQL** for the database
because it provided several pieces I expected to need in one platform:
authentication, PostgreSQL, Row Level Security (RLS), and pgvector for
later semantic retrieval.

For initial development, I used the Gemini API because its free tier 
allowed me to prototype the application without incurring generation costs. 
This was a development choice rather than a deployment assumption: the free 
tier is not appropriate for private company data, so a production deployment 
would require a paid API tier or another provider. I also designed the LLM 
interface to be provider-agnostic, allowing the underlying model or provider 
to be changed without restructuring the rest of the application.

The first version established a simple application data model in which 
Supabase Auth manages authenticated users, each user can own multiple 
conversations, and each conversation contains its associated messages.

I implemented email/password authentication and connected Supabase Auth
users to a public user profile table using the same UUID. This UUID 
provides a consistent identity across authentication, user profiles, 
conversations, and database access policies. I also added PostgreSQL 
privileges and RLS policies so users could only access conversations 
and messages belonging to their own UUID.

With the basic application and conversation storage working, I moved on 
to how conversation context would be managed for the LLM.

## 2. Adding conversation memory

Sending the entire conversation history to the LLM indefinitely is highly
inefficient, so I implemented a simple rolling summarization strategy. 
The 20 most recent messages are provided to the LLM, while older history
is compressed into a running summary. The full raw conversation is still
retained so they can still be displayed or used by future memory systems.

This was intentionally a lossy memory design. A summary can omit a small
detail that later becomes relevant. I considered semantic retrieval over
older conversation messages as a future improvement, but postponed it
because document RAG was a higher-priority problem and the same
retrieval ideas could eventually be reused for long-term conversational
memory.

## 3. Building the first RAG pipeline

The next major step was allowing the assistant to answer questions using
internal company documents. For the initial prototype, I created a small
test corpus of eight synthetic company documents covering areas such as HR 
policies, security procedures, and organizational information. I added 
`documents` and `document_chunks` tables 
and enabled pgvector in PostgreSQL. The ingestion pipeline supports Markdown, 
TXT, PDF, and DOCX files. Documents are split into overlapping chunks of 
approximately 250 words with 50 words of overlap. Each chunk is converted 
into a 768-dimensional Gemini embedding and stored in PostgreSQL. 

I chose 
Gemini Embedding 2 partly because the application was already using the Gemini 
API, which kept the initial RAG architecture simple. I kept the embedding and 
generation components separate, however, so the response-generating LLM could 
later be changed without requiring the document corpus to be re-embedded. 
I chose 768 dimensions rather than the model's larger default
representation because it reduces vector storage and similarity-search 
computation. The dimensionality can be tuned if later experiments show that 
retrieval quality requires a larger representation. 

The documents are ingested as follows:
```text
Company documents
        ↓
Text extraction
        ↓
Chunk documents
        ↓
Generate embeddings
        ↓
Store in database
```

At retrieval time, the user's question is converted into a 768-dimensional 
embedding using the same embedding model used during document ingestion. 
PostgreSQL and pgvector compare this query embedding against the stored 
document-chunk embeddings using cosine similarity and returns the most 
semantically similar chunks. Because the embedding vectors are already 
stored in Supabase, the similarity search is performed directly in the 
database rather than transferring the full embedding collection into Python 
for every request. The retrieved chunks are then provided to the LLM alongside
the user's question as context for generating a grounded response. 
This first implementation retrieved the five most semantically similar 
chunks and provided them to the LLM as context for generating its answer. 
The source filename for each chunk was also included so the assistant could
cite the relevant company documents.

## 4. Separating general questions from company retrieval

Once RAG was implemented, I had to handle the fact that every request did not 
actually need retrieval. A user might ask a general question such as a 
programming question, request writing help, or continue a normal conversation. 
I added an LLM router that runs before retrieval and classifies each request
into one of two categories:
- `GENERAL` — the question can be answered without internal company information,
  so document retrieval is skipped.
- `COMPANY_CONTEXT_REQUIRED` — the question depends on company-specific
  information, so the request is sent through the RAG pipeline.

This avoids unnecessary retrieval for general questions while ensuring
company-specific questions are grounded in internal documents.

While testing follow-up questions, I found a second problem. In a conversation 
regarding an employee probation period, the answering LLM received conversation
history and could understand a follow-up message such as "What happens during it?",
but because vector retrieval operated only on the latest user message, the embedding
model did not know what "it" referred to. To solve this, I made retrieval 
context-aware. The router received recent conversation context and, for 
company-specific follow-ups, produced a self-contained retrieval query. 
For example, the follow-up question "What happens during it?" was rewritten as 
"What happens during the employee probation period?"

The rewritten query is used only for embedding and retrieval. The
original user message and conversation history are still passed to the
final answering LLM. Routing and query rewriting were combined into the same 
LLM request so that context interpretation did not require another API call.

This later became an important architectural consideration because the LLM 
router was quite slow, leading me to investigate whether it could be 
replaced with a faster classifier. 

The answer generation pipeline is as follows:

```text
User message + conversation history
              ↓
          LLM Router
              ↓
    ┌─────────┴─────────────────────┐
    ↓                               ↓
 GENERAL                 COMPANY_CONTEXT_REQUIRED
    │                               ↓
    │                     Rewrite retrieval query
    │                               ↓
    │                         Embed query
    │                               ↓
    │                   Vector similarity search
    │                               ↓
    │                     Retrieve Top-K chunks
    │                               ↓
    │                 Combine conversation context 
    │                      and retrieved chunks
    │                               │
    └───────────────┬───────────────┘
                    ↓
               Answering LLM
                    ↓
                 Response
```

## 5. Adding permission-aware retrieval

For an internal company assistant, the user must also be authorized to see
the semantically best chunk. I assigned documents access levels such as:
-   `employee`
-   `manager`
-   `hr`
-   `admin`

The permission check occurs during retrieval. PostgreSQL filters
candidate documents according to the user's role before returning chunks
to the application. I considered more granular project- and team-level 
permissions, but postponed them as the role-based implementation was sufficient 
to test the architecture. Additional permissions could be implemented based 
on a company's specific requirements.

## 6. Adding observability and failure handling

Before systematically evaluating or scaling the system, I added instrumentation
so performance problems could be measured rather than guessed. The system records
metrics including:
- routing, embedding, vector search, retrieval, generation latency
- input and output token usage
- LLM call retries
- handled API errors
- estimated API cost

I also added retry handling for transient Gemini or API failures such as rate
limits and service errors using exponential backoff. Separating these 
measurements by pipeline stage became important later because latency could 
originate from several independent components rather than retrieval alone.

## 7. Building an evaluation harness

Once the core pipeline worked, manually asking questions in the
Stremlit interface was no longer sufficient. I built an extensive 
evaluation suite with questions related to the company documents and
general queries. The evaluator records multiple dimensions of system
behavior rather than reducing performance to a single answer score.

The main metrics include:

| Metric | Purpose |
|---|---|
| Router accuracy | Was the query routed correctly? |
| Retrieval accuracy | Was the expected document retrieved within the production Top-K retrieval set? |
| Recall@K | Was the expected document retrieved within the top K chunks? |
| Groundedness | Were company-specific claims supported by retrieved context? |
| Answer correctness | Did the final response answer the evaluation question? |
| Citation validity | Did cited sources actually come from retrieval? |
| Citation support | Did the cited documents support the associated claims? |
| Permission accuracy | Did retrieval respect the user's document permissions? |


I evaluate retrieval at several depths (Recall@5, Recall@10, Recall@20,
and Recall@50) while the production answer receives a smaller top-K
context. This distinction later became useful for separating candidate 
retrieval failures, where the expected document was absent even at larger 
retrieval depths, from ranking failures, where the expected document was 
retrieved but ranked too low to reach the answering LLM. K was initially 
chosen to be 10, so the top 10 documents were retrieved for context. 

The evaluator also uses an LLM judge for answer-level metrics. Manual
inspection later revealed that automated evaluation itself needs to be
audited: some test questions and reference answers were overly specific,
and a semantically valid answer could be marked wrong if it did not use
the arbitrarily designated source or include every expected detail.

As a result, I do not interpret answer correctness in isolation.
Retrieval recall, groundedness, citation support, and case-level failure
analysis are necessary to understand what actually failed.

## 8. Evaluating the initial 8-document system

I ran the 50-case evaluation suite against the
original eight-document system to establish an initial baseline.

| Metric | Result |
| --- | ---: |
| Router accuracy | 98% (49/50) |
| Retrieval accuracy* | 93.8% (30/32) |
| Answer correctness | 92% (46/50) |
| Groundedness | 100% (50/50) |
| Citation validity | 98% (49/50) |
| Citation support | 100% (50/50) |
| Permission accuracy | 100% (50/50) |
 **[Evaluation Results](evals/mvp/eval_results_v1.json)**

*Retrieval accuracy is calculated only across the 32 cases where retrieval
of a specific expected document was applicable.

The initial system performed well on the small corpus. The automated evaluation
measured 92% answer correctness (46/50), however
manual review showed that three of the four failed cases were evaluation 
ambiguities rather than clear factual errors. The assistant correctly excluded
restricted documents while still providing valid answers using relevant 
information from other documents the user was authorized to access. 
Because the evaluator expected answers based on the restricted documents, 
these valid responses were incorrectly marked as failures. Considering these
three evaluation false negatives, effective answer correctness could reasonably
be interpreted as up to **98%** (49/50), although I retain 92% (46/50) as the 
original automated evaluation score. This reveals a limitation in the 
evaluation design: valid, grounded answers from authorized sources can 
be incorrectly marked as failures when they differ from the predefined 
expected answer.

This established a useful baseline, but it did not demonstrate that the same
retrieval behavior would hold as the corpus grew. I therefore next scaled the
corpus to 1,000 documents and reran the evaluation.


## 9. Scaling from 8 to 1000 documents

To test the system at a more realistic scale, I expanded the corpus using
Confluence documents from [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench),
a synthetic benchmark designed to represent internal company knowledge. The
full benchmark contains more than 500,000 documents across nine enterprise
sources, including 5,189 Confluence pages. I used 1000 of these
Confluence documents for the first scaling experiment.

Scaling to 1000 documents immediately changed the engineering problem. 
The original ingestion pipeline had been designed for only a handful of 
documents, so sequential embedding and database operations became 
increasingly inefficient. I improved the ingestion pipeline to make 
it practical for larger document corpora. I replaced sequential database 
writes with bulk inserts to reduce database overhead, and added concurrent
embedding and document processing so multiple chunks and documents could be 
processed in parallel rather than one at a time. To make long ingestion runs 
more reliable, I added retries for transient API failures, along with 
checkpoint/resume behavior so interrupted 
runs could restart without reprocessing completed documents. I also added
progress and performance metrics to track ingestion throughput and identify
bottlenecks as the corpus scaled. 

Small experiments demonstrated the impact of these changes:

| Approach | Approximate time for 10 documents |
| --- | ---: |
| Bulk database inserts | 22.2 s |
| Concurrent embedding | 9.3 s |
| Concurrent document + embedding | 5.6 s |

When I tested approximately 100 documents, I began encountering Gemini 429 
rate-limit errors. I upgraded from the free API tier to Tier 1 to increase 
the available rate limits. I also considered using Google's asynchronous 
Batch API to process large numbers of embedding requests more efficiently 
and avoid relying on many concurrent synchronous requests, but ultimately 
did not implement it. The synchronous concurrent pipeline was already fast 
enough for the development corpus, and the Batch API could be added later 
if larger-scale ingestion required it.

The optimized pipeline successfully ingested 1,000 documents
in several minutes.

## 10. Evaluating the 1,000-document system

After ingesting 1,000 documents, I reran the evaluation suite to establish a
larger-corpus baseline before scaling further. The 1,000-document system achieved:

| Metric              | Result |
|---------------------|-------:|
| Router accuracy     |   100% |
| Retrieval accuracy  |    94% |
| Recall@5            |    94% |
| Recall@10           |    94% |
| Recall@20           |    94% |
| Recall@50           |    94% |
| Answer correctness  |    94% |
| Groundedness        |   100% |
| Citation validity   |   100% |
| Citation support    |   100% |
| Permission accuracy |   100% |


The flat Recall@K curve was particularly useful. When the expected document was
retrieved, it was generally already ranked within the first five results.
Increasing retrieval depth from 5 to 50 therefore provided no additional
document-level recall at this corpus size. This gave me a baseline against 
which I could measure what happened when the corpus grew even further.

## 11. Scaling from 1,000 to 5,000 documents: Adding an HNSW index

I next expanded the synthetic corpus from 1,000 to 5,000 documents, producing 
roughly 35,000 document chunks. The ingestion pipeline was capable of handling 
the larger corpus, but scaling exposed a different bottleneck: retrieval.

After ingesting the 5,000-document corpus, evaluation began failing immediately
with PostgreSQL:

```text
canceling statement due to statement timeout
code: 57014
```
The retrieval query ordered document chunks by cosine vector distance. At the
larger corpus size, the database could no longer reliably complete the search
within the configured statement timeout. To solve this retrieval at scale 
problem, I added a Hierarchical Navigable Small World (HNSW) index to the 
document-chunk embedding column in 
PostgreSQL using pgvector, which required no change to the Python retrieval 
logic. HNSW organizes similar embedding vectors into a graph, allowing the 
search to quickly navigate toward regions where the most relevant vectors 
are likely to be rather than comparing against every stored embedding. 
This makes retrieval much more efficient at scale, with the tradeoff that the
search is approximate and may occasionally miss a relevant chunk.

After the index was created, the same 5,000-document
evaluation completed without the previous vector-search timeout.
This was an important design decision because I had deliberately not added HNSW
to the original eight-document system. At that scale, it was unnecessary
complexity. I introduced it only after scaling demonstrated a concrete retrieval
bottleneck.

## 12. Evaluating the 5,000-document system

With retrieval working again, I reran the same evaluation framework against the
5000-document corpus. Comparing the larger corpus to the 1000 document corpus:

| Metric | 1k Documents | 5k Documents |
| --- |-------------:|-------------:|
| Router accuracy |         100% |         100% |
| Retrieval accuracy |          94% |          88% |
| Recall@5 |          94% |          80% |
| Recall@10 |          94% |          88% |
| Recall@20 |          94% |          90% |
| Recall@50 |          94% |          94% |
| Answer correctness |          94% |          88% |
| Groundedness |         100% |         100% |
| Citation validity |         100% |         100% |
| Citation support |         100% |         100% |
| Permission accuracy |         100% |         100% |

The results showed that retrieval remained effective as the corpus scaled, 
although ranking became more difficult as more semantically similar documents 
competed for the highest positions. Recall increased from 88% at K=10 to 94% 
at K=50, suggesting that most remaining retrieval failures were caused by 
relevant documents being ranked too low rather than being entirely absent
from the candidate set. At the same time, groundedness, citation support, 
and permission accuracy remained at 100%, indicating that the system continued
to generate well-supported responses from retrieved context and enforce access
controls at the larger scale.

Manual error analysis showed that the lower retrieval and answer-correctness 
scores at 5,000 documents did not have a single cause. The larger corpus 
introduced substantial semantic overlap, making it harder for dense retrieval 
to distinguish between closely related policies and systems. Some cases were
genuine retrieval failures where the required document or answer-bearing chunk
never reached the generator. Other cases were evaluation false negatives: the 
expected document was not retrieved, but other relevant documents contained 
valid information that still allowed the system to produce a correct, grounded
answer. The analysis also exposed failures at later stages of the pipeline. In
one case, the exact answer-bearing chunk was included in the LLM's retrieved 
context, but it appeared alongside several semantically similar chunks containing
different policies and values, and the generator failed to identify the 
correct evidence. 

Together, these cases showed that the aggregate scores reflect several distinct
issues, which are listed below along with possible solutions:

- **Document-level retrieval failures:** In some cases, the intended document was
  not retrieved within the top-10 results, even though it appeared in the top-50
  set. This could potentially be improved with reranking, hybrid keyword and 
 semantic search, or better query formulation.

- **Chunk-level retrieval failures:** Retrieving the correct document does not
  guarantee that the chunk containing the required evidence reaches the
  generator. In one case, a highly relevant chunk from the correct document was
  retrieved, while the separate chunk containing the actual answer ranked too
  low. Potential improvements include better chunking, reranking at the chunk 
  level, or hybrid search.

- **Candidate-retrieval failures:** In harder cases, the required document or
  answer-bearing evidence was absent even from the top-50 retrieved results. A
  reranker cannot solve this because it can only reorder candidates that were
  already retrieved. These failures may require improvements to query
  formulation, hybrid search, chunk representation, or metadata filtering.

- **Generation over competing context:** In at least one case, the exact
  answer-bearing chunk was included in the context sent to the LLM, but several
  other retrieved chunks described similar policies with conflicting values.
  The generator failed to identify and use the most directly relevant evidence.
  This could potentially be improved through better context selection,
  reranking, or stronger prompting around source specificity.

- **Evaluation ambiguity:** The synthetic corpus contains many overlapping
  policies with different valid values. Some evaluation questions were too broad
  to uniquely identify the document expected by the evaluator. As a result, the
  system could retrieve a different but highly relevant source and produce a
  valid, grounded answer while still being marked as a retrieval or answer
  failure. These cases suggest that the evaluation questions should be made more
  specific and that retrieval quality should not be measured solely by whether
  one predetermined document was returned.

- **Safe failures:** When the required evidence was genuinely unavailable to the
  generator, the model often stated that the information was unavailable rather
  than inventing an answer. This helps explain why answer correctness can fall
  below 100% while groundedness remains 100%.

## 16. Investigating latency

After retrieval quality reached a reasonable MVP point, latency became
the next obvious bottleneck.

The initial 5,000-document top-10 measurements were approximately:

  Stage                             p50
  ---------------------------- --------
  Router                         1.58 s
  Retrieval                      0.49 s
  Generation                     3.87 s
  Total measured AI pipeline     6.25 s

Retrieval was no longer the main latency problem. Generation dominated,
with routing adding another substantial sequential LLM request.

This changed the optimization priority. Further pgvector tuning was
unlikely to make the application feel dramatically faster when
generation alone took several seconds.

I expanded observability to separate embedding time from vector-search
time and to record stage-level tokens, retries, errors, and cost.

A later evaluation unexpectedly showed both routing and generation at
roughly twice their earlier latency while retrieval remained almost
unchanged. Because the instrumentation around the Gemini calls was
simple and retrieval timing was stable, I investigated whether the
slowdown came from retries, timing bugs, network conditions, or the
model/API itself.

A minimal repeated Gemini test using the prompt `Reply with exactly: OK`
still produced calls in roughly the 1.5--3.4 second range with **zero
retries and zero handled errors** in the observed samples. That
supported the conclusion that the remote model/API latency itself was a
meaningful contributor rather than the new observability code.

This investigation is still ongoing. The important architectural result
is already clear: the largest optimization opportunity has shifted from
vector retrieval to the **LLM stages**, especially generation and
potentially the LLM-based router.

## 17. The router is now an optimization target

The router currently performs two related tasks in one LLM call:

1.  classify the request as general or company-context-required
2.  rewrite context-dependent company questions into self-contained
    retrieval queries

The evaluation suite has shown excellent routing accuracy, but the
router adds a sequential model call before retrieval.

A faster rules-based, embedding-based, or learned classifier could
reduce latency and cost. However, replacing the LLM router is not just a
classification experiment because the current call also performs
contextual query rewriting.

A meaningful comparison therefore needs to evaluate more than router
accuracy. Future router experiments should measure:

-   route classification accuracy
-   downstream Recall@K
-   answer correctness
-   latency
-   token usage and cost

Possible designs include a lightweight classifier for routing while
invoking rewriting only for company follow-ups that actually need
conversational reference resolution.

## 18. Frontend performance is a separate problem

The Streamlit application also exhibits latency on interactions such as
login and starting a new chat, which cannot be explained by Gemini
generation.

Streamlit reruns the application script on interactions, and the current
app restores the Supabase auth session, loads the user's role, and
fetches conversation metadata during reruns.

This means there are now two different performance questions:

-   **ML pipeline latency:** routing, embedding, retrieval, generation
-   **application/UI latency:** Streamlit reruns, authentication, and
    database requests

I considered replacing the frontend with React/Next.js and exposing the
existing Python system through FastAPI. That would provide more control
over application responsiveness, but I postponed the rewrite because the
project's primary goal is to demonstrate ML and backend engineering.
Most of the existing Python modules could remain unchanged if the
frontend is replaced later.

## 19. Where the system stands

The project began as a basic authenticated chatbot and evolved through a
sequence of measured engineering problems:

``` text
Persistent chat
    ↓
Conversation memory
    ↓
Document RAG
    ↓
Context-aware query rewriting
    ↓
Permission-aware retrieval
    ↓
Evaluation
    ↓
Scalable ingestion
    ↓
5K-document retrieval timeout
    ↓
HNSW indexing
    ↓
Citation redesign
    ↓
1K vs. 5K scaling experiment
    ↓
Failure analysis
    ↓
Stage-level observability
    ↓
Latency/model optimization
```

The most useful lesson from the project has been that the interesting
engineering decisions were rarely the features I could have added in
advance. They came from measuring actual failures.

I did not add HNSW until vector search timed out. I did not add a
reranker simply because modern RAG systems often use one. I did not
adopt asynchronous batch ingestion when synchronous concurrency was
sufficient. I did not rewrite the frontend before determining whether
frontend engineering was the most valuable next step.

The current system is therefore intentionally not the most complicated
architecture possible. It is a measured baseline with known strengths,
known failure modes, and instrumentation that makes future changes
testable.

## 20. Next experiments

The next phase is focused less on adding features and more on testing
targeted improvements:

-   benchmark alternative routing approaches while preserving contextual
    query rewriting
-   compare generator models on quality, latency, and cost
-   reduce generation prompt/context size where possible
-   evaluate reranking or hybrid retrieval specifically against
    demonstrated retrieval failures
-   improve evidence-level retrieval metrics beyond document-level
    Recall@K
-   produce final latency and cost comparisons
-   improve Streamlit responsiveness or replace the frontend after the
    ML system is stable
-   revisit richer project/team permissions and long-term conversational
    retrieval if they become important to the target use case

The goal for each change is the same: establish a baseline, identify a
specific limitation, make one targeted change, and measure whether it
actually improves the system.

## TODO: COST ANALYSIS 