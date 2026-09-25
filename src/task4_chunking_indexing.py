"""
Task 4 — Chunking, embedding và indexing.

Pipeline: đọc Markdown trong data/standardized/ -> chunk -> embed -> upsert ChromaDB.

Quyết định thiết kế:
    - Metadata lấy từ header do Task 3 sinh ra (**Source:**), nên `url` và `title`
      là URL/tiêu đề thật của tài liệu gốc chứ không phải tên file. Citation nhờ
      vậy trỏ thẳng về trang AWS.
    - Header truy vết bị cắt khỏi content trước khi chunk, tránh việc chunk đầu
      của mỗi tài liệu chỉ chứa boilerplate.
    - Mỗi chunk giữ thêm `section` (heading Markdown gần nhất) để câu trả lời chỉ
      được đúng mục chứa thông tin, đúng yêu cầu của bài toán.
    - ID chunk = "<doc id>::chunk-<index>" nên ổn định giữa các lần chạy; kết hợp
      upsert và xoá chunk mồ côi để index lại không nhân bản dữ liệu.
    - Embedding dùng API Gemini (gemini-embedding-001, 3072 chiều) theo .env, nên
      không phải tải model nặng về máy. Provider local vẫn được giữ để chạy offline.

Task 5 phải import lại chính embed_texts() ở đây để query và corpus cùng model,
cùng dimension.

Chạy:
    python -m src.task4_chunking_indexing
"""

import os
import re
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
# Giữ giá trị starter: tài liệu AWS là văn bản thủ tục, nhiều bước ngắn và bảng,
# nên chunk 500 ký tự đủ giữ trọn một bước mà vẫn định vị chính xác; overlap 50
# nối phần tiếp giáp giữa hai bước liền nhau.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or "gemini"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"

# Dimension mặc định theo model; dùng để cảnh báo khi vector trả về khác dự kiến.
DEFAULT_EMBEDDING_DIMS = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "BAAI/bge-m3": 1024,
}
EMBEDDING_DIM = DEFAULT_EMBEDDING_DIMS.get(EMBEDDING_MODEL, 0)

COLLECTION_NAME = "rag_documents"

# Chroma giới hạn số bản ghi mỗi lần ghi.
UPSERT_BATCH_SIZE = 1000

# Gemini giới hạn số text mỗi request; batch nhỏ cũng dễ retry hơn khi bị 429.
API_EMBED_BATCH_SIZE = 50
API_MAX_ATTEMPTS = 6

# Header do Task 3 sinh ra kết thúc ở dòng "---" đầu tiên.
HEADER_SEPARATOR = "\n---\n"
TITLE_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
SOURCE_PATTERN = re.compile(r"^\*\*Source:\*\*\s*(\S.*?)\s*$", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


# ---------------------------------------------------------------- embedding


@lru_cache(maxsize=1)
def _sentence_transformer():
    """Load model một lần cho cả tiến trình; load lại rất tốn thời gian."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _gemini_client():
    """Tạo client Gemini một lần; key đọc từ .env, không hard-code."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu GEMINI_API_KEY trong .env (EMBEDDING_PROVIDER=gemini cần key này)"
        )
    return genai.Client(api_key=api_key)


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text


def _is_daily_quota(error: Exception) -> bool:
    """Hết quota theo ngày (quotaId ...PerDay...): chờ bao lâu cũng không hồi phục."""
    return "PerDay" in str(error)


def _call_with_retry(operation, label: str):
    """Gọi API embedding, backoff khi bị rate limit.

    Free tier của Gemini giới hạn token mỗi phút, nên corpus vài trăm chunk rất
    dễ gặp 429; retry để pipeline không chết giữa đường.
    """
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            return operation()
        except Exception as error:
            last = attempt == API_MAX_ATTEMPTS - 1
            # Hết quota ngày thì báo lỗi ngay: retry chỉ làm mỗi query treo ~155s
            # trước khi Task 9 lùi về BM25.
            if last or not _is_rate_limit(error) or _is_daily_quota(error):
                raise
            delay = 5 * 2**attempt
            print(f"  {label}: rate limit, chờ {delay}s rồi thử lại...")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng provider khai báo trong .env.

    Task 4 dùng cho corpus, Task 5 dùng cho query — chung một hàm nên không thể
    lệch model hoặc dimension.
    """
    if not texts:
        return []

    provider = EMBEDDING_PROVIDER.strip().lower()

    if provider == "sentence_transformers":
        model = _sentence_transformer()
        vectors = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        model_name = EMBEDDING_MODEL if "/" not in EMBEDDING_MODEL else "text-embedding-3-small"
        vectors: list[list[float]] = []
        for batch in _batched(texts, API_EMBED_BATCH_SIZE):
            response = client.embeddings.create(model=model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    if provider == "gemini":
        client = _gemini_client()
        model_name = EMBEDDING_MODEL if "/" not in EMBEDDING_MODEL else "gemini-embedding-001"
        batches = list(_batched(texts, API_EMBED_BATCH_SIZE))
        vectors: list[list[float]] = []

        for number, batch in enumerate(batches, 1):
            if len(batches) > 1:
                print(f"  embedding batch {number}/{len(batches)} ({len(batch)} texts)")
            response = _call_with_retry(
                lambda batch=batch: client.models.embed_content(
                    model=model_name, contents=batch
                ),
                label=f"batch {number}",
            )
            vectors.extend(list(item.values) for item in response.embeddings)

        return vectors

    raise ValueError(
        f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER!r} "
        "(dùng sentence_transformers | openai | gemini)"
    )


# ---------------------------------------------------------------- vectorstore


@lru_cache(maxsize=1)
def _chroma_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    return _chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _to_chroma_metadata(metadata: dict) -> dict:
    """Chroma chỉ nhận scalar; bỏ giá trị None (ví dụ url của tài liệu không có URL)."""
    return {key: value for key, value in metadata.items() if value is not None}


def _from_chroma_metadata(metadata: dict | None) -> dict:
    """Khôi phục `url=None` đã bị lược khi ghi, để đúng contract."""
    restored = dict(metadata or {})
    restored.setdefault("url", None)
    return restored


# ---------------------------------------------------------------- load & chunk


def _parse_standardized(text: str) -> tuple[str | None, str | None, str]:
    """Tách header của Task 3 thành (title, url, body)."""
    head, separator, body = text.partition(HEADER_SEPARATOR)
    if not separator:
        return None, None, text.strip()

    title_match = TITLE_PATTERN.search(head)
    source_match = SOURCE_PATTERN.search(head)

    url = source_match.group(1) if source_match else None
    if url and not url.lower().startswith(("http://", "https://")):
        url = None

    return (title_match.group(1) if title_match else None, url, body.strip())


def load_documents() -> list[dict]:
    """Đọc Markdown trong data/standardized/ và trả về danh sách Document."""
    documents: list[dict] = []

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        title, url, body = _parse_standardized(path.read_text(encoding="utf-8"))
        if not body:
            print(f"Skip (empty): {path.name}")
            continue

        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": body,
                "metadata": {
                    "source": path.name,
                    "title": (title or path.stem).strip(),
                    "doc_type": "legal" if "legal" in path.parts else "news",
                    "url": url,
                },
            }
        )

    return documents


def _split_by_section(content: str, default_section: str) -> list[tuple[str, str]]:
    """Cắt document thành các khối (section, text) theo heading Markdown.

    PDF sau khi convert gần như không còn heading; khi đó cả document là một khối
    mang tên tiêu đề tài liệu.
    """
    blocks: list[tuple[str, list[str]]] = []
    current = default_section

    for line in content.splitlines():
        heading = HEADING_PATTERN.match(line)
        if heading:
            current = heading.group(2).strip() or current
            continue
        if not blocks or blocks[-1][0] != current:
            blocks.append((current, []))
        blocks[-1][1].append(line)

    return [(section, "\n".join(lines).strip()) for section, lines in blocks]


@lru_cache(maxsize=1)
def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định, chunk_index và section."""
    splitter = _splitter()
    chunks: list[dict] = []

    for document in documents:
        metadata = document["metadata"]
        default_section = metadata.get("title") or metadata.get("source") or ""
        index = 0

        for section, block in _split_by_section(document["content"], default_section):
            for text in splitter.split_text(block):
                text = text.strip()
                if not text:
                    continue
                chunks.append(
                    {
                        "id": f"{document['id']}::chunk-{index}",
                        "content": text,
                        "metadata": {
                            **metadata,
                            "chunk_index": index,
                            "section": section,
                        },
                    }
                )
                index += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên các field khác."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB và xoá chunk mồ côi của lần index trước."""
    collection = get_collection()

    for batch in _batched(chunks, UPSERT_BATCH_SIZE):
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_to_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )

    # Tài liệu bị rút ngắn hoặc gỡ bỏ sẽ để lại chunk cũ; xoá để store khớp corpus.
    current_ids = {chunk["id"] for chunk in chunks}
    stale = sorted(set(collection.get(include=[])["ids"]) - current_ids)
    if stale:
        for batch in _batched(stale, UPSERT_BATCH_SIZE):
            collection.delete(ids=batch)
        print(f"Removed {len(stale)} stale chunks")


def load_chunks_from_store() -> list[dict]:
    """Đọc lại chunk đã index, để BM25 (Task 6) dùng đúng corpus với dense."""
    stored = get_collection().get(include=["documents", "metadatas"])
    chunks = [
        {
            "id": chunk_id,
            "content": content,
            "metadata": _from_chroma_metadata(metadata),
        }
        for chunk_id, content, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"]
        )
    ]
    # Chroma không đảm bảo thứ tự; sort để corpus ổn định giữa các lần chạy.
    chunks.sort(
        key=lambda item: (
            item["metadata"].get("source", ""),
            item["metadata"].get("chunk_index", 0),
        )
    )
    return chunks


def _reset_collection_if_dimension_changed(dimension: int) -> None:
    """Đổi model embedding thì vector cũ khác dimension và Chroma sẽ báo lỗi.

    Xoá collection cũ để index lại từ đầu, thay vì để pipeline chết giữa upsert.
    """
    collection = get_collection()
    if collection.count() == 0:
        return

    existing = collection.peek(limit=1).get("embeddings")
    if existing is None or len(existing) == 0:
        return

    old_dimension = len(existing[0])
    if old_dimension == dimension:
        return

    print(
        f"Dimension đổi {old_dimension} -> {dimension} (model embedding khác); "
        f"xoá collection {COLLECTION_NAME} và index lại."
    )
    _chroma_client().delete_collection(COLLECTION_NAME)


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    if not chunks:
        print("Không có chunk nào — chạy Task 3 trước.")
        return

    print(f"Embedding bằng {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}")
    embedded_chunks = embed_chunks(chunks)
    dimension = len(embedded_chunks[0]["embedding"])
    if EMBEDDING_DIM and dimension != EMBEDDING_DIM:
        print(f"Chú ý: dimension thực tế {dimension} khác dự kiến {EMBEDDING_DIM}")

    _reset_collection_if_dimension_changed(dimension)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks (dim={dimension}) into {CHROMA_DIR}")
    print(f"Collection now holds {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
