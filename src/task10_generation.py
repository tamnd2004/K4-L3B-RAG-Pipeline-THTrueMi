"""
Task 10 — Generation có citation.

Luồng:
    1. Retrieve top-k chunks (Task 9).
    2. Đánh số citation [n] theo thứ hạng trong `sources` (n = vị trí + 1).
    3. Reorder để giảm lost-in-the-middle; số citation đi theo chunk nên không đổi.
    4. Format context kèm title/section/source, gọi LLM theo LLM_PROVIDER.
    5. Kiểm tra citation: bỏ số không tồn tại; không còn citation hợp lệ nào thì
       trả safe refusal với sources=[] và retrieval_source="none".

Vì sao đánh số trước khi reorder: nếu gán nhãn theo vị trí sau reorder thì nhãn
[2] trong context trỏ tới chunk khác với sources[1] mà UI hiển thị — citation
không còn truy ngược được về SearchResult.

Chạy:
    python -m src.task10_generation
"""

import logging
import os
import re
import time

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1024

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Free tier trả 429 khi vượt quota, model mới hay trả 503 khi quá tải.
LLM_MAX_ATTEMPTS = 4
LLM_RETRY_BASE_SECONDS = 2

logger = logging.getLogger(__name__)

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = f"""You are a lookup assistant for AWS migration guidance, serving cloud architects, migration leads and migration engineers.

Mandatory rules:
1. Use only the information in the provided context passages. No outside knowledge, no guessing.
2. Each context passage starts with a label [n]. After every sentence or bullet that uses the context, cite it as [n]; for several passages write [1][3]. Only use numbers that exist in the context.
3. Reply in the language named at the end of the user message. Be concise and direct; use bullets for steps or conditions.
4. If the context answers only part of the question, you must answer that part with citations and state plainly which part the sources do not cover. Never attach a citation to a statement that the sources do not cover something.
5. Only when no passage is relevant to any part of the question, reply with exactly this sentence and nothing else, untranslated: "{SAFE_REFUSAL}"
6. Do not invent URLs or document names; sources are shown to the user separately."""

# Chữ cái chỉ có trong tiếng Việt; dùng để chọn ngôn ngữ trả lời một cách tất định
# thay vì để model tự đoán (model hay trả tiếng Việt cho câu hỏi tiếng Anh).
VIETNAMESE_CHARS = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)

# Khớp "[1]", "[1, 3]" và "[1,3]".
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

RETRYABLE_MARKERS = ("429", "503", "resource_exhausted", "unavailable", "overloaded", "rate limit")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context, không sửa list đầu vào.

    Input xếp theo độ liên quan giảm dần. Xen kẽ đầu/cuối: [0, 1, 2, 3, 4] ->
    [0, 2, 4, 3, 1], nên hai chunk mạnh nhất nằm ở hai mép context.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có nhãn citation, title, section và source cho từng đoạn.

    Nhãn lấy từ `citation` của chunk (số thứ tự trong sources) nếu có, nếu không
    thì dùng vị trí trong list.
    """
    parts = []
    for position, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        label = chunk.get("citation", position)
        header = f"[{label}] Title: {metadata['title']}"
        section = metadata.get("section")
        if section and section != metadata["title"]:
            header += f" | Section: {section}"
        header += f" | Source: {metadata['source']}"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def _is_retryable(error: Exception) -> bool:
    text = str(error)
    # Hết quota ngày (quotaId ...PerDay...) thì retry vô ích.
    if "PerDay" in text:
        return False
    return any(marker in text.lower() for marker in RETRYABLE_MARKERS)


def _call_provider(provider: str, model: str, system_prompt: str, user_message: str) -> str:
    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content

    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config={
                "system_instruction": system_prompt,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "automatic_function_calling": {"disable": True},
            },
        )
        return response.text

    if provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; expected openai, gemini, or anthropic"
    )


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình; trả về text thuần."""
    provider = LLM_PROVIDER.strip().lower()
    model = LLM_MODEL.strip()
    if not model:
        raise RuntimeError("LLM_MODEL is not configured")

    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            text = _call_provider(provider, model, system_prompt, user_message)
            break
        except Exception as error:
            if attempt == LLM_MAX_ATTEMPTS - 1 or not _is_retryable(error):
                raise
            delay = LLM_RETRY_BASE_SECONDS * 2**attempt
            logger.warning("%s tạm lỗi (%s), thử lại sau %ss", provider, error, delay)
            time.sleep(delay)

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"{provider} returned an empty response")
    return text.strip()


def _clean_citations(answer: str, source_count: int) -> tuple[str, list[int]]:
    """Bỏ số citation không có trong sources; trả về (answer, các số hợp lệ đã dùng)."""
    used: list[int] = []

    def replace(match: re.Match) -> str:
        valid = [
            number
            for number in (int(part) for part in match.group(1).split(","))
            if 1 <= number <= source_count
        ]
        for number in valid:
            if number not in used:
                used.append(number)
        # \x00 đánh dấu citation bị bỏ để xoá luôn khoảng trắng đứng trước nó.
        return "".join(f"[{number}]" for number in valid) or "\x00"

    cleaned = CITATION_PATTERN.sub(replace, answer)
    return re.sub(r"[ \t]*\x00", "", cleaned).strip(), used


def _refusal() -> dict:
    return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}


def _retrieval_source(chunks: list[dict]) -> str:
    """retrieval_source mô tả đường tạo ra sources.

    Đường sản phẩm (generate_with_citation) chỉ cho hybrid hoặc pageindex. Giá trị
    "dense" chỉ xuất hiện ở Config A của A/B evaluation, không đi ra UI.
    """
    methods = {chunk.get("retrieval_method") for chunk in chunks}
    if methods == {"pageindex"}:
        return "pageindex"
    if methods == {"dense"}:
        return "dense"
    return "hybrid"


def answer_with_sources(query: str, chunks: list[dict]) -> tuple[dict, str]:
    """Sinh câu trả lời từ chunks đã retrieve; trả về (GenerationResult, status).

    Tách khỏi generate_with_citation để A/B evaluation dùng đúng một prompt và
    generator cho cả hai cấu hình retrieval. status là một trong: answered,
    no_chunks, llm_error, refused, uncited.
    """
    if not chunks:
        return _refusal(), "no_chunks"

    # Gắn số citation trên bản sao; sources trả ra vẫn là chunk gốc, đúng thứ tự.
    numbered = [{**chunk, "citation": number} for number, chunk in enumerate(chunks, 1)]
    context = format_context(reorder_for_llm(numbered))
    language = "Vietnamese" if VIETNAMESE_CHARS.search(query) else "English"
    user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer in {language}."

    try:
        raw_answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:
        logger.warning("Generation failed; returning safe refusal: %s", error)
        return _refusal(), "llm_error"

    answer, cited = _clean_citations(raw_answer, len(chunks))
    if not cited:
        # Không truy ngược được về nguồn nào thì không đưa cho người dùng.
        status = "refused" if SAFE_REFUSAL in raw_answer else "uncited"
        if status == "uncited":
            logger.warning("Câu trả lời không có citation hợp lệ, trả safe refusal")
        return _refusal(), status

    return {
        "answer": answer,
        "sources": list(chunks),
        "retrieval_source": _retrieval_source(chunks),
    }, "answered"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult: answer, sources và retrieval_source đồng bộ nhau."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return _refusal()

    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as error:
        logger.warning("Retrieval failed; returning safe refusal: %s", error)
        return _refusal()

    result, _ = answer_with_sources(query, chunks)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for demo_query in (
        "Trước khi cutover cần kiểm tra những gì, và khi nào phải rollback?",
        "Công thức nấu phở bò?",
    ):
        result = generate_with_citation(demo_query)
        print(f"\n=== {demo_query}\n[{result['retrieval_source']}] {result['answer']}")
        for number, source in enumerate(result["sources"], 1):
            metadata = source["metadata"]
            print(f"  [{number}] {metadata['title']} — {metadata.get('section', '')} ({source['score']:.4f})")
