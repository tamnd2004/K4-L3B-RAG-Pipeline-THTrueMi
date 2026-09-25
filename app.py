"""
Chatbot tra cứu kiến thức migration lên AWS.

UI gọi generate_with_citation(query, top_k) và chỉ hiển thị nguồn lấy từ
`sources` của GenerationResult: số [n] trong câu trả lời là phần tử thứ n của
sources, nên người dùng mở đúng đoạn và đúng URL mà model đã đọc. URL lấy từ
metadata của chunk, không bao giờ lấy từ văn bản model sinh ra.

Chạy:
    streamlit run app.py
"""

import time

import streamlit as st
from dotenv import load_dotenv

from src.task4_chunking_indexing import EMBEDDING_MODEL
from src.task9_retrieval_pipeline import SCORE_THRESHOLD
from src.task10_generation import (
    CITATION_PATTERN,
    LLM_MODEL,
    LLM_PROVIDER,
    generate_with_citation,
)


load_dotenv()

EXAMPLE_QUESTIONS = [
    "Ứng dụng nên rehost hay replatform?",
    "Trước khi xếp ứng dụng vào một wave cần thu thập metadata gì?",
    "Migrate PostgreSQL lên Aurora PostgreSQL có những cách nào?",
    "When is the T-1 go/no-go meeting held and who attends?",
    "Khi nào cần rollback sau cutover và xử lý dữ liệu thế nào?",
]

SOURCE_LABELS = {
    "hybrid": ("Hybrid · Dense + BM25 + RRF", "blue"),
    "pageindex": ("PageIndex fallback", "violet"),
    "none": ("Không có nguồn phù hợp", "gray"),
}

DOC_TYPE_LABELS = {"legal": "Playbook PDF", "news": "Trang hướng dẫn"}


st.set_page_config(page_title="AWS Migration Assistant", page_icon="☁️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []


def cited_numbers(answer: str) -> set[int]:
    """Các số [n] xuất hiện trong câu trả lời."""
    numbers: set[int] = set()
    for match in CITATION_PATTERN.finditer(answer):
        numbers.update(int(part) for part in match.group(1).split(","))
    return numbers


def score_label(source: dict) -> str:
    """Nói rõ thang đo: RRF chỉ phản ánh thứ hạng, không phải xác suất."""
    method = source.get("retrieval_method")
    score = source.get("score", 0.0)
    if method == "hybrid":
        return f"RRF {score:.4f}"
    if method == "dense":
        return f"cosine {score:.4f}"
    if method == "pageindex":
        return f"rank score {score:.2f}"
    return f"score {score:.4f}"


def render_sources(message: dict, message_index: int) -> None:
    """Hiển thị sources của một câu trả lời; dùng được cho cả lịch sử."""
    retrieval_source = message.get("retrieval_source", "none")
    sources = message.get("sources") or []
    label, color = SOURCE_LABELS.get(retrieval_source, (retrieval_source, "gray"))

    meta = [f"{len(sources)} nguồn", f"top_k={message.get('top_k')}"]
    if message.get("latency") is not None:
        meta.append(f"{message['latency']:.1f}s")

    badge_col, meta_col = st.columns([0.45, 0.55])
    with badge_col:
        st.badge(label, color=color)
    meta_col.caption(" · ".join(meta))

    if not sources:
        st.info(
            "Không tìm thấy bằng chứng đủ tin cậy trong bộ tài liệu, nên chatbot "
            "không trả lời thay vì đoán."
        )
        return

    cited = cited_numbers(message["content"])
    st.markdown("**Nguồn đã đưa vào context**")

    for number, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("title", "Không rõ tiêu đề")
        section = metadata.get("section")
        is_cited = number in cited

        with st.container(border=True):
            text_col, link_col = st.columns([0.8, 0.2])
            with text_col:
                marker = "✅ được trích dẫn" if is_cited else "không được trích"
                st.markdown(f"**[{number}] {title}**")
                details = []
                if section and section != title:
                    details.append(f"Mục: {section}")
                details.append(DOC_TYPE_LABELS.get(metadata.get("doc_type"), metadata.get("doc_type", "")))
                details.append(score_label(source))
                details.append(marker)
                st.caption(" · ".join(detail for detail in details if detail))
            url = metadata.get("url")
            if url:
                link_col.link_button("Mở nguồn ↗", url, key=f"link-{message_index}-{number}")

            with st.expander("Xem đoạn được dùng"):
                st.caption(f"{metadata.get('source', '')} · chunk #{metadata.get('chunk_index', '?')}")
                st.markdown(source.get("content", ""))


with st.sidebar:
    st.title("☁️ AWS Migration Assistant")
    st.caption(
        "Trả lời câu hỏi về migration lên AWS từ 3 playbook large migration và 5 "
        "trang hướng dẫn của AWS Prescriptive Guidance. Mỗi ý có citation [n] trỏ "
        "tới nguồn n bên dưới câu trả lời."
    )
    top_k = st.slider("Số chunk đưa vào context (top_k)", 3, 10, 5)

    st.markdown("**Câu hỏi mẫu**")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"example-{example}", use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    st.caption(
        f"Generator: `{LLM_PROVIDER}/{LLM_MODEL}`  \n"
        f"Embedding: `{EMBEDDING_MODEL}`  \n"
        f"Fallback threshold (cosine): `{SCORE_THRESHOLD}`"
    )
    if st.button("Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("Tra cứu kiến thức migration lên AWS")
st.caption(
    "Hỏi về rehost/replatform, xếp wave, migrate máy chủ, PostgreSQL, shared file "
    "system, cutover và rollback. Kiểm tra nguồn trước khi đưa ra quyết định."
)

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message, index)

query = st.chat_input("Nhập câu hỏi về migration lên AWS...")
if not query:
    query = st.session_state.pop("pending_query", None)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        with st.spinner("Đang tìm nguồn và soạn câu trả lời..."):
            try:
                result = generate_with_citation(query, top_k)
                error = None
            except Exception as exc:  # UI không được crash vì provider lỗi.
                result = {"answer": "", "sources": [], "retrieval_source": "none"}
                error = exc

        if error is not None:
            answer = "Hệ thống đang gặp lỗi khi xử lý câu hỏi, vui lòng thử lại."
            st.error(f"{answer} ({type(error).__name__})")
        else:
            answer = result["answer"]
            st.markdown(answer)

        message = {
            "role": "assistant",
            "content": answer,
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "top_k": top_k,
            "latency": time.perf_counter() - started,
        }
        render_sources(message, len(st.session_state.messages))

    st.session_state.messages.append(message)
