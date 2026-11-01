import re
import sys
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# Project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.graph.research_graph import ResearchGraph
from backend.database.feedback_store import (
    initialize_database,
    save_feedback,
    get_feedback_summary,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AdaptiveRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        [data-testid="stChatMessage"] {
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
        }

        [data-testid="stChatInput"] {
            max-width: 1050px;
            margin-left: auto;
            margin-right: auto;
        }

        .app-subtitle {
            color: #888888;
            font-size: 0.95rem;
            margin-top: -12px;
            margin-bottom: 25px;
        }

        .source-label {
            color: #888888;
            font-size: 0.8rem;
            margin-bottom: 4px;
        }

        .status-text {
            color: #888888;
            font-size: 0.85rem;
        }

        div.stButton > button {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load backend
# --------------------------------------------------

@st.cache_resource
def load_research_graph():
    initialize_database()
    return ResearchGraph()


research_graph = load_research_graph()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = None


# --------------------------------------------------
# Math formatting
# --------------------------------------------------

def render_answer(answer):
    """
    Convert common LaTeX delimiters into Streamlit-supported
    Markdown math delimiters before displaying the answer.
    """

    if not answer:
        st.write("No answer was generated.")
        return

    answer = str(answer)

    # Display math: \[ ... \] -> $$ ... $$
    answer = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "\n$$\n" + match.group(1).strip() + "\n$$\n",
        answer,
        flags=re.DOTALL,
    )

    # Inline math: \( ... \) -> $ ... $
    answer = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: "$" + match.group(1).strip() + "$",
        answer,
        flags=re.DOTALL,
    )

    # Handle equations surrounded by square brackets, for example:
    # [ \text{sim}(q,p)=E_Q(q)^\top E_P(p) ]
    

    answer = re.sub(
    r"(?m)^\s*\[\s*((?=[^\]]*\\(?:text|mathbb|top|in|sum|frac|mathbf))[^]]+)\]\s*[,.;]?\s*$",
    lambda match: "\n$$\n" + match.group(1).strip() + "\n$$\n",
    answer,
)

    st.markdown(answer)


# --------------------------------------------------
# Source display
# --------------------------------------------------

def display_sources(sources, message_id):
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            title = source.get("paper_title", "Research paper")
            section = source.get("section", "Unknown section")
            citation = source.get("citation", f"[{index}]")

            page_start = source.get("page_start")
            page_end = source.get("page_end")

            if page_start == page_end and page_start is not None:
                page_text = f"Page {page_start}"
            elif page_start is not None and page_end is not None:
                page_text = f"Pages {page_start}–{page_end}"
            else:
                page_text = "Page unavailable"

            st.markdown(f"**{citation} {title}**")
            st.caption(f"{section} · {page_text}")

            if index != len(sources):
                st.divider()


# --------------------------------------------------
# Retrieval inspection
# --------------------------------------------------

def display_retrieval_details(chunks, message_id):
    if not chunks:
        return

    with st.expander("🔍 Inspect retrieval and ranking"):
        st.caption(
            "These are the passages retrieved by the system. "
            "Feedback adjustments show how historical feedback affected ranking."
        )

        for index, chunk in enumerate(chunks, start=1):
            paper = (
                chunk.get("paper_title")
                or chunk.get("paper_id")
                or "Unknown paper"
            )

            section = chunk.get("section", "Unknown section")

            with st.expander(f"{index}. {paper} — {section}"):
                text_content = chunk.get(
                    "text",
                    chunk.get("content", "Passage text unavailable"),
                )
                st.write(text_content)

                original = chunk.get(
                    "original_score",
                    chunk.get("score"),
                )
                adjustment = chunk.get("feedback_adjustment")
                adjusted = chunk.get("adjusted_score")
                evidence = chunk.get("feedback_evidence_count", 0)

                metrics = st.columns(4)

                if original is not None:
                    metrics[0].metric("Similarity", f"{original:.4f}")

                if adjustment is not None:
                    metrics[1].metric(
                        "Feedback adjustment",
                        f"{adjustment:+.4f}",
                    )

                if adjusted is not None:
                    metrics[2].metric("Adjusted score", f"{adjusted:.4f}")

                metrics[3].metric("Feedback records", str(evidence))

                if adjustment is not None and abs(adjustment) > 0:
                    st.caption(
                        "Historical feedback influenced this passage's ranking."
                    )
                else:
                    st.caption(
                        "No qualifying historical feedback influenced this passage."
                    )

                st.caption(f"Chunk ID: {chunk.get('chunk_id', 'Unavailable')}")


# --------------------------------------------------
# Feedback
# --------------------------------------------------

def display_feedback(message):
    answer_id = message.get("answer_id")

    if not answer_id:
        return

    feedback_state = message.get("feedback_state", "not_submitted")

    if feedback_state == "submitted":
        st.caption("Thanks for your feedback! ❤️")
        return

    if feedback_state == "negative_pending":
        reason_key = f"reason_{answer_id}"

        reason = st.selectbox(
            "What could be improved?",
            options=[
                "irrelevant_retrieval",
                "missing_information",
                "answer_incorrect",
            ],
            format_func=lambda value: {
                "irrelevant_retrieval": "Retrieved passages were irrelevant",
                "missing_information": "Important information was missing",
                "answer_incorrect": "The answer was incorrect",
            }[value],
            key=reason_key,
        )

        if st.button("Submit feedback", key=f"submit_{answer_id}"):
            try:
                save_feedback(
                    answer_id=answer_id,
                    rating=-1,
                    reason=reason,
                )
                message["feedback_state"] = "submitted"
                st.rerun()
            except Exception as error:
                st.error(f"Could not save feedback: {error}")

        return

    col1, col2, _ = st.columns([0.6, 0.6, 8])

    with col1:
        if st.button("👍", key=f"like_{answer_id}", help="Helpful answer"):
            try:
                save_feedback(
                    answer_id=answer_id,
                    rating=1,
                    reason=None,
                )
                message["feedback_state"] = "submitted"
                st.rerun()
            except Exception as error:
                st.error(f"Could not save feedback: {error}")

    with col2:
        if st.button("👎", key=f"dislike_{answer_id}", help="Not helpful"):
            message["feedback_state"] = "negative_pending"
            st.rerun()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("📚 AdaptiveRAG")
    st.caption("Your research paper assistant")

    st.divider()

    st.subheader("Knowledge base")

    papers = [
        ("RAG", "Retrieval-Augmented Generation"),
        ("DPR", "Dense Passage Retrieval"),
        ("ColBERT", "Contextualized Late Interaction"),
        ("HyDE", "Hypothetical Document Embeddings"),
        ("RAGAS", "RAG Evaluation Framework"),
    ]

    for short_name, description in papers:
        with st.container():
            st.markdown(f"**{short_name}**")
            st.caption(description)

    st.divider()

    st.subheader("Feedback overview")

    try:
        summary = get_feedback_summary()

        st.metric(
            "Answers generated",
            summary.get("total_answers", 0),
        )

        st.metric(
            "Feedback received",
            summary.get("total_feedback", 0),
        )

    except Exception:
        st.caption("Feedback statistics unavailable.")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_feedback = None
        st.rerun()

    st.caption("Powered by FastEmbed · Qdrant · Groq · LangGraph")


# --------------------------------------------------
# Main chat header
# --------------------------------------------------

st.title("Research Assistant")
st.markdown(
    '<p class="app-subtitle">'
    'Ask questions about research papers and explore the evidence behind answers.'
    '</p>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Welcome screen
# --------------------------------------------------

if not st.session_state.messages:
    st.markdown("### What would you like to understand today?")
    st.write(
        "Ask questions about RAG, dense retrieval, ColBERT, HyDE, "
        "or RAG evaluation."
    )

    suggestions = [
        "How does Dense Passage Retrieval work?",
        "How does ColBERT improve document retrieval?",
        "Explain the HyDE approach.",
        "What is the purpose of RAGAS?",
    ]

    cols = st.columns(2)

    for index, suggestion in enumerate(suggestions):
        with cols[index % 2]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion
                st.rerun()


# --------------------------------------------------
# Conversation history
# --------------------------------------------------

for message_index, message in enumerate(st.session_state.messages):

    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])

    else:
        with st.chat_message("assistant", avatar="📚"):

            render_answer(message.get("answer", ""))

            retry_count = message.get("retry_count", 0)

            if retry_count > 0:
                st.caption(
                    f"🔄 A corrective retrieval retry was performed "
                    f"({retry_count} retry)."
                )

            display_sources(
                message.get("sources", []),
                message.get("answer_id", str(message_index)),
            )

            display_feedback(message)

            display_retrieval_details(
                message.get("retrieved_chunks", []),
                message.get("answer_id", str(message_index)),
            )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

typed_question = st.chat_input(
    "Ask a question about your research papers..."
)

pending_question = st.session_state.pop("pending_question", None)

question = typed_question or pending_question

if question and question.strip():

    question = question.strip()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.spinner("Searching research papers and preparing your answer..."):
        try:
            result = research_graph.ask(question)

            assistant_message = {
                "role": "assistant",
                "answer": result.get("answer", ""),
                "answer_id": result.get("answer_id"),
                "sources": result.get("sources", []),
                "retrieved_chunks": result.get("retrieved_chunks", []),
                "retry_count": result.get("retry_count", 0),
                "selected_grade": result.get("selected_grade", {}),
                "feedback_state": "not_submitted",
            }

            st.session_state.messages.append(assistant_message)

        except Exception as error:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "answer": (
                        "I encountered an error while processing your question. "
                        "Please try again."
                    ),
                    "error": str(error),
                    "feedback_state": "not_submitted",
                }
            )

    st.rerun()