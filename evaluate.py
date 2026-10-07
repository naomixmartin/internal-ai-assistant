import json
import re
from llm import generate_response
from retrieval import retrieve_chunks, build_document_context
from router import route_query
import os
import numpy as np
from observability import estimate_cost


EVAL_FILE = "evals/scaled/eval_cases.json"
RESULTS_FILE = "evals/scaled/eval_results_gemini_3.8_5k_files_top10_v2.json"

RECALL_K_VALUES = [5, 10, 20, 50]
ANSWER_MATCH_COUNT = 10


def generate_answer(question, route, retrieved_chunks):
    source_map = {}

    if route == "COMPANY_CONTEXT_REQUIRED":
        document_context, source_map = build_document_context(retrieved_chunks)

        messages = [{"role": "user", "content": f"""
            Relevant company information:
            
            {document_context}
            
            Use this information when it is relevant to the user's question.
            Cite factual claims from company documents using the numbered source
            that supports the claim, for example [1].
            
            Only cite source numbers provided in the company information.
            Do not include filenames or the word "SOURCE" inside citations.
            
            If the company information does not contain enough information to answer
            the question, say that you could not find the answer in the company documents.
            """
            },
            {"role": "user", "content": question
            }
        ]
        contents = messages
    else:
        contents = question

    answer, metadata = generate_response(contents, stage="evaluation", return_metadata=True)
    return answer, source_map, metadata


def get_cited_source_numbers(answer):
    return sorted(set(int(number) for number in re.findall(r"\[(\d+)\]", answer)))


def judge_answer(case, answer, retrieved_chunks):
    retrieved_context, _ = build_document_context(retrieved_chunks)

    prompt = f"""
    You are evaluating an internal company AI assistant.
    
    Evaluate the assistant answer using the test case and retrieved context below.
    
    Question:
    {case["question"]}
    
    User role:
    {case["role"]}
    
    Should this question be answerable from information available to this user?
    {case["answerable"]}
    
    Reference answer:
    {case["reference_answer"]}
    
    Retrieved company context:
    {retrieved_context if retrieved_context else "NONE"}
    
    Assistant answer:
    {answer}
    
    Score these fields:
    
    1. correct
    - If answerable is true, true only if the assistant gives the materially correct answer.
    - The wording does not need to match the reference answer.
    - If reference_answer is null, judge whether the response appropriately fulfills the question.
    - If answerable is false, true only if the assistant does not provide unavailable or restricted company information and appropriately says the information is unavailable when company information would be required.
    
    2. grounded
    - For company-specific factual claims, true only if they are supported by the retrieved company context.
    - For GENERAL questions that do not require company context, true if the answer does not invent company-specific facts.
    
    3. citations_supported
    - For COMPANY_CONTEXT_REQUIRED questions with an answer supported by company documents, true only if cited company documents actually support the claims attributed to them.
    - If the assistant appropriately says the answer cannot be found and makes no company-document factual claim, true.
    - For GENERAL questions, true.
    
    Return only valid JSON in exactly this format:
    {{
      "correct": true,
      "grounded": true,
      "citations_supported": true
    }}
    """

    response = generate_response(prompt, stage="evaluation").strip()

    # tolerate markdown code fences if the model adds them
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

    return json.loads(response)


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total_cases = len(cases)

    # continue results file if a previous run failed
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    completed_ids = {result["id"] for result in results}

    for case in cases:
        if case["id"] in completed_ids:
            print(f"Skipping completed case {case['id']}/{total_cases}")
            continue

        print(f'Running case {case["id"]}/{total_cases}: {case["question"]}')

        # route the question
        actual_route, retrieval_query, routing_latency_ms, routing_metadata = route_query(case["question"], "")
        route_pass = actual_route == case["expected_route"]

        # always retrieve for evaluation so retrieval is measured independently of routing
        evaluation_chunks, evaluation_retrieval_metadata = retrieve_chunks( retrieval_query, case["role"], match_count=max(RECALL_K_VALUES))

        # retrieve top-10 chunks for production only when company context is needed
        retrieved_chunks = []
        retrieval_metadata = {"retrieval_latency_ms": 0, "embedding_latency_ms": 0, "vector_search_latency_ms": 0}

        if actual_route == "COMPANY_CONTEXT_REQUIRED":
            retrieved_chunks, retrieval_metadata = retrieve_chunks(retrieval_query, case["role"], match_count=ANSWER_MATCH_COUNT)

        retrieved_documents = {chunk["filename"] for chunk in evaluation_chunks[:ANSWER_MATCH_COUNT]}

        # expected-document retrieval score
        expected_documents = set(case["expected_documents"])
        min_expected = case["min_expected_documents"]

        # calculate whether the expected document appears within each retrieval depth
        recall_at_k = {}

        for k in RECALL_K_VALUES:
            documents_at_k = {chunk["filename"] for chunk in evaluation_chunks[:k]}
            recall_at_k[str(k)] = bool(expected_documents & documents_at_k)

        if min_expected > 0:
            expected_found = len(expected_documents & retrieved_documents)
            retrieval_pass = expected_found >= min_expected
        else:
            retrieval_pass = None

        # permission score: forbidden documents must never be retrieved
        forbidden_documents = set(case["forbidden_documents"])
        permission_violations = forbidden_documents & retrieved_documents
        permission_pass = len(permission_violations) == 0

        # generate the actual assistant answer
        answer, source_map, generation_metadata = generate_answer(case["question"], actual_route, retrieved_chunks)
        generation_latency_ms = generation_metadata["generation_latency_ms"]

        # calculate production end-to-end latency
        total_latency_ms = (routing_latency_ms + retrieval_metadata["retrieval_latency_ms"] + generation_latency_ms)

        # calculate costs
        routing_cost_usd = estimate_cost(routing_metadata["model"], routing_metadata["input_tokens"], routing_metadata["output_tokens"])
        generation_cost_usd = estimate_cost(generation_metadata["model"], generation_metadata["input_tokens"], generation_metadata["output_tokens"])
        input_tokens = ((routing_metadata["input_tokens"] or 0) + (generation_metadata["input_tokens"] or 0))
        output_tokens = ((routing_metadata["output_tokens"] or 0) + (generation_metadata["output_tokens"] or 0))
        total_tokens = ((routing_metadata["total_tokens"] or 0) + (generation_metadata["total_tokens"] or 0))
        estimated_cost_usd = (routing_cost_usd + generation_cost_usd if routing_cost_usd is not None and generation_cost_usd is not None else None)

        # deterministic citation check: cited docs must have been retrieved
        cited_source_numbers = get_cited_source_numbers(answer)
        invalid_citations = {number for number in cited_source_numbers if number not in source_map}
        cited_documents = {source_map[number] for number in cited_source_numbers if number in source_map}

        if actual_route == "GENERAL":
            citation_present_pass = True
            citation_retrieval_pass = len(cited_documents) == 0

        elif case["answerable"]:
            citation_present_pass = len(cited_documents) > 0
            citation_retrieval_pass = len(invalid_citations) == 0

        else:
            citation_present_pass = True
            citation_retrieval_pass = len(invalid_citations) == 0

        # use an llm judge for semantic answer quality
        judge = judge_answer(case, answer, retrieved_chunks)

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "role": case["role"],
            "expected_route": case["expected_route"],
            "actual_route": actual_route,
            "route_pass": route_pass,
            "retrieval_query": retrieval_query,
            "expected_documents": sorted(expected_documents),
            "retrieved_documents": sorted(retrieved_documents),
            "retrieved_chunks": [
                {"id": chunk["id"],
                 "filename": chunk["filename"],
                 "chunk_index": chunk["chunk_index"],
                 "content": chunk["content"],
                 "similarity": chunk["similarity"]} for chunk in retrieved_chunks],
            "evaluation_chunks": [
                {"rank": i + 1,
                 "filename": chunk["filename"],
                 "chunk_index": chunk["chunk_index"],
                 "similarity": chunk["similarity"],
                 "content": chunk["content"]} for i, chunk in enumerate(evaluation_chunks)],
            "retrieval_pass": retrieval_pass,
            "recall_at_k": recall_at_k,
            "routing_metrics": {
                "provider": routing_metadata["provider"],
                "model": routing_metadata["model"],
                "input_tokens": routing_metadata["input_tokens"],
                "output_tokens": routing_metadata["output_tokens"],
                "total_tokens": routing_metadata["total_tokens"],
                "latency_ms": routing_latency_ms,
                "retry_count": routing_metadata["retry_count"],
                "errors_handled": routing_metadata["errors_handled"],
                "cost_usd": routing_cost_usd
            },
            "retrieval_metrics": {
                "embedding_latency_ms": retrieval_metadata["embedding_latency_ms"],
                "vector_search_latency_ms": retrieval_metadata["vector_search_latency_ms"],
                "retrieval_latency_ms": retrieval_metadata["retrieval_latency_ms"]
            },
            "generation_metrics": {
                "provider": generation_metadata["provider"],
                "model": generation_metadata["model"],
                "input_tokens": generation_metadata["input_tokens"],
                "output_tokens": generation_metadata["output_tokens"],
                "total_tokens": generation_metadata["total_tokens"],
                "latency_ms": generation_latency_ms,
                "retry_count": generation_metadata["retry_count"],
                "errors_handled": generation_metadata["errors_handled"],
                "cost_usd": generation_cost_usd
            },
            "total_metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": total_latency_ms,
                "cost_usd": estimated_cost_usd
            },
            "evaluation_retrieval_metrics": evaluation_retrieval_metadata,
            "forbidden_documents": sorted(forbidden_documents),
            "permission_violations": sorted(permission_violations),
            "permission_pass": permission_pass,
            "answer": answer,
            "cited_documents": sorted(cited_documents),
            "invalid_citations": sorted(invalid_citations),
            "citation_retrieval_pass": citation_retrieval_pass,
            "answer_correct": judge["correct"],
            "grounded": judge["grounded"],
            "citations_supported": judge["citations_supported"],
            "citation_present_pass": citation_present_pass,
        }

        results.append(result)

        # save progress after every completed case
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # calculate summary metrics
    total = len(results)
    route_correct = sum(r["route_pass"] for r in results)
    retrieval_results = [r for r in results if r["retrieval_pass"] is not None]
    retrieval_correct = sum(r["retrieval_pass"] for r in retrieval_results)
    permission_correct = sum(r["permission_pass"] for r in results)
    answer_correct = sum(r["answer_correct"] for r in results)
    grounded_correct = sum(r["grounded"] for r in results)
    citation_retrieval_correct = sum(r["citation_retrieval_pass"] for r in results)
    citation_present_correct = sum(r["citation_present_pass"] for r in results)
    citation_support_correct = sum( r["citations_supported"] for r in results)
    recall_results = {k: sum(r["recall_at_k"][str(k)] for r in results) for k in RECALL_K_VALUES}
    latency_fields = {
        "router": ("routing_metrics", "latency_ms"),
        "embedding": ("retrieval_metrics", "embedding_latency_ms"),
        "vector_search": ("retrieval_metrics", "vector_search_latency_ms"),
        "retrieval": ("retrieval_metrics", "retrieval_latency_ms"),
        "generation": ("generation_metrics", "latency_ms"),
        "total": ("total_metrics", "latency_ms")
    }

    latency_summary = {}
    for stage, (section, field) in latency_fields.items():
        values = [r[section][field] for r in results]
        latency_summary[stage] = {"p50": np.percentile(values, 50), "p95": np.percentile(values, 95)}

    # summarize retries and handled llm errors
    routing_retries = sum(r["routing_metrics"]["retry_count"] for r in results)
    generation_retries = sum(r["generation_metrics"]["retry_count"] for r in results)
    routing_errors = sum(len(r["routing_metrics"]["errors_handled"]) for r in results)
    generation_errors = sum(len(r["generation_metrics"]["errors_handled"]) for r in results)

    print("\n--- Evaluation Results ---")
    print(f"Router accuracy: "
          f"{route_correct}/{total} "
          f"({route_correct / total * 100:.1f}%)")

    if retrieval_results:
        print(f"Retrieval accuracy: "
              f"{retrieval_correct}/{len(retrieval_results)} "
              f"({retrieval_correct / len(retrieval_results) * 100:.1f}%)")

    print("\nLatency (ms)")
    print(f"{'Stage':<12} {'p50':>10} {'p95':>10}")
    for stage in latency_fields:
        print(
            f"{stage.capitalize():<12} "
            f"{latency_summary[stage]['p50']:>10.1f} "
            f"{latency_summary[stage]['p95']:>10.1f}"
        )

    print("\nLLM retries/errors")
    print(f"Router retries: {routing_retries}")
    print(f"Generation retries: {generation_retries}")
    print(f"Router errors handled: {routing_errors}")
    print(f"Generation errors handled: {generation_errors}")

    for k in RECALL_K_VALUES:
        recall_correct = recall_results[k]
        print(f"Recall@{k}: "
              f"{recall_correct}/{total} "
              f"({recall_correct / total * 100:.1f}%)")

    print(f"Permission accuracy: "
          f"{permission_correct}/{total} "
          f"({permission_correct / total * 100:.1f}%)")

    print(f"Answer correctness: "
          f"{answer_correct}/{total} "
          f"({answer_correct / total * 100:.1f}%)")

    print(f"Groundedness: "
          f"{grounded_correct}/{total} "
          f"({grounded_correct / total * 100:.1f}%)")

    print(f"Citation retrieval validity: "
          f"{citation_retrieval_correct}/{total} "
          f"({citation_retrieval_correct / total * 100:.1f}%)")

    print(f"Citation presence: "
          f"{citation_present_correct}/{total} "
          f"({citation_present_correct / total * 100:.1f}%)")

    print(f"Citation support: "
          f"{citation_support_correct}/{total} "
          f"({citation_support_correct / total * 100:.1f}%)")

    # show failed cases
    print("\n--- Failed Cases ---")

    for result in results:
        failures = []

        if not result["route_pass"]:
            failures.append("router")

        if result["retrieval_pass"] is False:
            failures.append("retrieval")

        if not result["permission_pass"]:
            failures.append("permissions")

        if not result["answer_correct"]:
            failures.append("answer")

        if not result["grounded"]:
            failures.append("groundedness")

        if not result["citation_retrieval_pass"]:
            failures.append("citation retrieval")

        if not result["citation_present_pass"]:
            failures.append("citation missing")

        if not result["citations_supported"]:
            failures.append("citation support")

        if failures:
            print(f'Case {result["id"]}: {", ".join(failures)}')

    # save detailed results for later inspection
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
