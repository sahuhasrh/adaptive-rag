
import json
import os
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

from backend.retrieval.retriever import ResearchRetriever
from backend.database.feedback_store import (
    get_connection,
    initialize_database,
    normalize_query,
)

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


class ResearchState(TypedDict, total=False):
    question: str
    search_query: str
    first_chunks: list[dict]
    second_chunks: list[dict]
    selected_chunks: list[dict]
    first_grade: dict
    second_grade: dict
    selected_grade: dict
    retry_count: int
    answer: str
    answer_id: str


class ResearchGraph:
    def __init__(self):
        self.retriever = ResearchRetriever()
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

        workflow = StateGraph(ResearchState)

        workflow.add_node("retrieve_first", self.retrieve_first)
        workflow.add_node("grade_first", self.grade_first)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve_second", self.retrieve_second)
        workflow.add_node("grade_second", self.grade_second)
        workflow.add_node("select_first", self.select_first)
        workflow.add_node("select_best", self.select_best)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("retrieve_first")

        workflow.add_edge("retrieve_first", "grade_first")

        workflow.add_conditional_edges(
            "grade_first",
            self.route_after_first_grade,
            {
                "answer": "select_first",
                "retry": "rewrite_query",
            },
        )

        workflow.add_edge("rewrite_query", "retrieve_second")
        workflow.add_edge("retrieve_second", "grade_second")
        workflow.add_edge("grade_second", "select_best")
        workflow.add_edge("select_first", "generate")
        workflow.add_edge("select_best", "generate")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()

    
    def _grade_context(self, question: str, chunks: list[dict]) -> dict:
        if not chunks:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "No relevant passages were retrieved.",
            }

        context = "\n\n".join(
            f"Passage {i + 1}: {chunk['text']}"
            for i, chunk in enumerate(chunks)
        )

        try:
            response = self.llm.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You evaluate retrieved research-paper passages. "
                            "Decide whether the passages contain enough relevant "
                            "information to answer the question. Do not use "
                            "outside knowledge. Return ONLY valid JSON in this "
                            "exact format: "
                            '{"sufficient": true, "confidence": 0.9, '
                            '"reason": "Short explanation"}. '
                            "The confidence must be a numeric value between "
                            "0.0 and 1.0, never a word. Be conservative."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Retrieved passages:\n{context}"
                        ),
                    },
                ],
            )

            raw = response.choices[0].message.content.strip()

            # Handle accidental Markdown code fences.
            raw = raw.replace("```json", "").replace("```", "").strip()

            result = json.loads(raw)

            return {
                "sufficient": bool(result.get("sufficient", False)),
                "confidence": max(
                    0.0, min(1.0, float(result.get("confidence", 0.0)))
                ),
                "reason": str(result.get("reason", "")),
            }

        except Exception as error:
            print(f"Context grading failed: {error}")

            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "Context grading failed; treating evidence as insufficient.",
            }


    def retrieve_first(self, state: ResearchState):
        chunks = self.retriever.search_adaptive(
            state["question"], top_k=5
        )
        return {
            "search_query": state["question"],
            "first_chunks": chunks,
            "retry_count": 0,
        }

    def grade_first(self, state: ResearchState):
        grade = self._grade_context(
            state["question"], state.get("first_chunks", [])
        )
        return {"first_grade": grade}

    def route_after_first_grade(self, state: ResearchState):
        if state["first_grade"]["sufficient"]:
            return "answer"
        return "retry"

    def rewrite_query(self, state: ResearchState):
        response = self.llm.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite research questions to improve semantic "
                        "retrieval from academic papers. Preserve the "
                        "original intent. Use specific technical terms. "
                        "Return only the rewritten question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question: {state['question']}\n"
                        f"Why retrieval was insufficient: "
                        f"{state['first_grade']['reason']}\n"
                        "Rewrite the question for a better retrieval attempt."
                    ),
                },
            ],
        )

        rewritten = response.choices[0].message.content.strip()
        return {
            "search_query": rewritten or state["question"],
            "retry_count": 1,
        }

    def retrieve_second(self, state: ResearchState):
        chunks = self.retriever.search_adaptive(
            state["search_query"], top_k=5
        )
        return {"second_chunks": chunks}

    def grade_second(self, state: ResearchState):
        grade = self._grade_context(
            state["question"], state.get("second_chunks", [])
        )
        return {"second_grade": grade}

    def select_first(self, state: ResearchState):
        return {
            "selected_chunks": state.get("first_chunks", []),
            "selected_grade": state["first_grade"],
        }

    def select_best(self, state: ResearchState):
        first_grade = state["first_grade"]
        second_grade = state["second_grade"]

        # Prefer sufficient evidence. If both are sufficient or both
        # insufficient, use the higher context-grader confidence.
        if second_grade["sufficient"] and not first_grade["sufficient"]:
            use_second = True
        elif first_grade["sufficient"] and not second_grade["sufficient"]:
            use_second = False
        else:
            use_second = (
                second_grade["confidence"] > first_grade["confidence"]
            )

        if use_second:
            return {
                "selected_chunks": state.get("second_chunks", []),
                "selected_grade": second_grade,
            }

        return {
            "selected_chunks": state.get("first_chunks", []),
            "selected_grade": first_grade,
        }

    def generate(self, state: ResearchState):
        chunks = state.get("selected_chunks", [])
        grade = state.get("selected_grade", {})

        if not grade.get("sufficient", False):
            answer = (
                "I couldn't find enough relevant evidence in the available "
                "research papers to answer this question reliably. "
                "Try narrowing the question or asking about a specific "
                "concept from the papers."
            )
        else:
            context_parts = []
            for i, chunk in enumerate(chunks, start=1):
                context_parts.append(
                    f"[{i}] Paper: {chunk.get('paper_title')}\n"
                    f"Section: {chunk.get('section')}\n"
                    f"Pages: {chunk.get('page_start')}-"
                    f"{chunk.get('page_end')}\n"
                    f"Content: {chunk.get('text')}"
                )

            response = self.llm.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research-paper assistant. Answer "
                            "using only the supplied passages. Cite claims "
                            "using passage numbers such as [1] and [2]. "
                            "If a detail is not supported by the passages, "
                            "say so. Do not invent facts or citations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n\n"
                            f"Evidence:\n{chr(10).join(context_parts)}"
                        ),
                    },
                ],
            )
            answer = response.choices[0].message.content.strip()

        answer_id = str(uuid.uuid4())
        initialize_database()

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO answers (
                    answer_id,
                    original_query,
                    normalized_query,
                    retrieval_strategy,
                    retrieved_chunks_json,
                    relevance_scores_json,
                    answer_text,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    answer_id,
                    state["question"],
                    normalize_query(state["question"]),
                    "adaptive_langgraph",
                    json.dumps([c["chunk_id"] for c in chunks]),
                    json.dumps([c["score"] for c in chunks]),
                    answer,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        return {
            "answer": answer,
            "answer_id": answer_id,
        }

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({"question": question})

        sources = []
        for i, chunk in enumerate(result.get("selected_chunks", []), start=1):
            sources.append({
                "citation": f"[{i}]",
                "paper_title": chunk.get("paper_title"),
                "paper_id": chunk.get("paper_id"),
                "section": chunk.get("section"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "chunk_id": chunk.get("chunk_id"),
                "score": chunk.get("score"),
            })

        return {
            "answer_id": result["answer_id"],
            "answer": result["answer"],
            "sources": sources,
            "retrieved_chunks": result.get("selected_chunks", []),
            "retry_count": result.get("retry_count", 0),
            "selected_grade": result.get("selected_grade", {}),
        }
