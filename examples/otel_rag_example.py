"""
genai-telemetry over OpenTelemetry — RAG example.

Uses your existing genai-telemetry decorators with the OTLP exporter, so spans
flow:  app -> OTLPExporter -> OpenTelemetry Collector (:4318) -> Splunk / any OTel backend.

Prereqs:
    pip install genai-telemetry openai            # plus pinecone if you wire real retrieval
    # an OpenTelemetry Collector listening on :4318 (forwarding to Splunk genai:otel).
    # Use collector/otel-collector-config.yaml from the splunk app package.

    $env:OPENAI_API_KEY = "sk-..."                 # PowerShell
    python otel_rag_example.py
"""

import os
from typing import List

from openai import OpenAI

from genai_telemetry import (
    setup_telemetry,
    trace_llm,
    trace_embedding,
    trace_retrieval,
    trace_chain,
)

# --- send telemetry via OpenTelemetry (OTLP) ---------------------------------
setup_telemetry(
    workflow_name="support_qa",
    exporter="otlp",                              # <-- the OTel path
    otlp_endpoint="http://localhost:4318",        # your running collector
    service_name="support-assistant",
)

client = OpenAI()  # reads OPENAI_API_KEY from the environment


@trace_embedding(model="text-embedding-3-small")
def embed(text: str) -> List[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


@trace_retrieval(vector_store="pinecone", embedding_model="text-embedding-3-small")
def retrieve(query_vector: List[float]) -> List[str]:
    # Plug your real Pinecone query here; placeholder docs for the example.
    # from pinecone import Pinecone
    # res = index.query(vector=query_vector, top_k=5, include_metadata=True)
    # return [m["metadata"]["text"] for m in res["matches"]]
    return ["billing policy doc", "refund policy doc"]


# extract_content=True -> return the FULL response (so tokens are captured),
# but the decorator hands back just the content string.
@trace_llm(model_name="gpt-4o-mini", model_provider="openai", extract_content=True)
def generate(question: str, context: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Use this context:\n{context}"},
            {"role": "user", "content": question},
        ],
    )


@trace_chain(name="support_qa")
def answer(question: str) -> str:
    query_vector = embed(question)
    docs = retrieve(query_vector)
    return generate(question, "\n".join(docs))


if __name__ == "__main__":
    for q in ["How do refunds work?", "What payment methods are supported?"]:
        print(f"\nQ: {q}\nA: {answer(q)}")
    print("\nSent via OTLP. If the collector forwards to Splunk:  index=genai_traces | head 20")
