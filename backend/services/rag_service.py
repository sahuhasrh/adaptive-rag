
import os
import uuid

from groq import Groq

from backend.retrieval.retriever import ResearchRetriever
from backend.database.feedback_store import (
    initialize_database,
    record_answer,
)


SYSTEM_PROMPT = """
You are AdaptiveRAG, a research paper assistant.

Answer the user's question using ONLY the supplied research paper context.

Rules:
1. Do not invent facts or citations.
2. If the context does not contain the answer, clearly say so.
3. Explain technical concepts in simple, understandable language.
4. Use information from multiple papers when relevant.
5. Cite factual claims using [1], [2], etc.
6. Do not confuse a paper's findings with your own interpretation.
7. Never fabricate page numbers, paper titles, or sections.

At the end, provide a Sources section listing the citations used.
"""


class RAGService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")

        initialize_database()

        self.client = Groq(api_key=api_key)
        self.model = os.getenv(
            "LLM_MODEL",
            "llama-3.3-70b-versatile",
        )
        self.retriever = ResearchRetriever()

    def answer(self, question: str, top_k: int = 5) -> dict:
        answer_id = str(uuid.uuid4())

        results = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        if not results:
            answer_text = (
                "I could not find relevant information in the "
                "available research papers."
            )

            record_answer(
                answer_id=answer_id,
                query=question,
                answer_text=answer_text,
                retrieved_chunks=[],
            )

            return {
                "answer_id": answer_id,
                "answer": answer_text,
                "sources": [],
                "retrieved_chunks": [],
            }

        context_parts = []
        sources = []

        for index, result in enumerate(results, start=1):
            sources.append({
                "citation_id": index,
                "paper_title": result["paper_title"],
                "paper_id": result["paper_id"],
                "section": result["section"],
                "page_start": result["page_start"],
                "page_end": result["page_end"],
                "score": result["score"],
            })

            context_parts.append(
                f"""
[SOURCE {index}]
Paper: {result['paper_title']}
Section: {result['section']}
Pages: {result['page_start']}-{result['page_end']}

Content:
{result['text']}
"""
            )

        context = "\n\n".join(context_parts)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
Research context:

{context}

Question:
{question}

Answer using the supplied context and cite sources.
""",
                },
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        answer_text = response.choices[0].message.content or ""

        record_answer(
            answer_id=answer_id,
            query=question,
            answer_text=answer_text,
            retrieved_chunks=results,
            retrieval_strategy="baseline_vector",
        )

        return {
            "answer_id": answer_id,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": results,
        }
