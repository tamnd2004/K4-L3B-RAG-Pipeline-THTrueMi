"""
A/B evaluation: Config A (dense-only) vs Config B (hybrid dense + BM25 + RRF).

Hai config chỉ khác retrieval strategy (`use_reranking` của Task 9). Golden
dataset, generator, prompt (answer_with_sources của Task 10), top_k, threshold
và evaluator giữ nguyên.

Giai đoạn:
    generate  chạy pipeline cho mọi golden case ở cả hai config, xen kẽ A/B theo
              từng câu để rate limit/độ trễ mạng rơi đều lên hai bên; lưu answer,
              contexts, trạng thái và latency vào results/generations.json.
    score     chấm 4 metric RAGAS (faithfulness, answer relevancy, context recall,
              context precision) bằng evaluator LLM khác generator; lưu
              results/scores.json và results/summary.json.

Chạy từ gốc repository:
    python group_project/evaluation/run_evaluation.py            # cả hai giai đoạn
    python group_project/evaluation/run_evaluation.py generate
    python group_project/evaluation/run_evaluation.py score
"""

import asyncio
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.task4_chunking_indexing import EMBEDDING_MODEL, get_collection  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve  # noqa: E402
from src.task10_generation import LLM_MODEL, LLM_PROVIDER, TEMPERATURE, answer_with_sources  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
GENERATIONS_PATH = RESULTS_DIR / "generations.json"
SCORES_PATH = RESULTS_DIR / "scores.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
# ".cache/" đã nằm trong .gitignore.
CACHE_DIR = EVAL_DIR / ".cache" / "ragas"

TOP_K = 5
CONFIGS = {
    "A": {"name": "dense-only", "use_reranking": False},
    "B": {"name": "hybrid + RRF", "use_reranking": True},
}

# Evaluator khác generator (gemini-3.5-flash-lite) để giảm thiên lệch tự chấm, và
# có quota free tier riêng nên chấm điểm không làm cạn quota của chatbot.
# gemini-3.6-flash bị loại: free tier chỉ 20 request/ngày, trong khi một lần chấm
# 2 config x 15 case x 4 metric cần khoảng 300 request.
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL") or "gemini-3.1-flash-lite"
# Embedding chỉ dùng để chấm answer relevancy, giống hệt nhau cho A và B. Tách
# khỏi embedding của retriever (gemini-embedding-001) để judge không dùng chung
# không gian vector với hệ thống bị chấm; quota free tier cũng tính riêng theo model.
EVAL_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
METRIC_CONCURRENCY = 3
METRIC_MAX_ATTEMPTS = 10

logger = logging.getLogger("evaluation")


# ------------------------------------------------------------------ helpers


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class _FallbackRecorder(logging.Handler):
    """Bắt log của Task 9 để biết case nào đã thử PageIndex fallback."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# ------------------------------------------------------------------ generate


def run_generation() -> dict:
    golden = _read_json(GOLDEN_PATH)
    pipeline_logger = logging.getLogger("src.task9_retrieval_pipeline")
    pipeline_logger.setLevel(logging.INFO)
    recorder = _FallbackRecorder()
    pipeline_logger.addHandler(recorder)

    # Warm-up không gọi API: mở Chroma và build BM25 index một lần, để case đầu
    # của Config B không phải gánh chi phí khởi tạo.
    get_collection().count()
    lexical_search("warm up", top_k=1)

    records = {key: [] for key in CONFIGS}
    for number, case in enumerate(golden, 1):
        golden_sources = sorted({Path(item["file"]).name for item in case.get("citations", [])})
        for key, config in CONFIGS.items():
            recorder.messages.clear()
            started = time.perf_counter()
            chunks = retrieve(case["question"], top_k=TOP_K, use_reranking=config["use_reranking"])
            retrieved_at = time.perf_counter()
            result, status = answer_with_sources(case["question"], chunks)
            finished = time.perf_counter()

            retrieved_sources = [chunk["metadata"]["source"] for chunk in chunks]
            records[key].append(
                {
                    "id": case["id"],
                    "retrieval_type": case.get("retrieval_type"),
                    "question": case["question"],
                    "expected_answer": case["expected_answer"],
                    "golden_sources": golden_sources,
                    "answer": result["answer"],
                    "status": status,
                    "retrieval_source": result["retrieval_source"],
                    "contexts": [chunk["content"] for chunk in chunks],
                    "retrieved": [
                        {
                            "id": chunk["id"],
                            "source": chunk["metadata"]["source"],
                            "section": chunk["metadata"].get("section"),
                            "method": chunk["retrieval_method"],
                            "score": chunk["score"],
                        }
                        for chunk in chunks
                    ],
                    "source_hit": any(source in golden_sources for source in retrieved_sources),
                    "fallback_attempted": any("fallback" in message for message in recorder.messages),
                    "context_chars": sum(len(chunk["content"]) for chunk in chunks),
                    "latency_retrieval_s": round(retrieved_at - started, 3),
                    "latency_generation_s": round(finished - retrieved_at, 3),
                    "latency_total_s": round(finished - started, 3),
                }
            )
            print(
                f"[{number:02d}/{len(golden)}] {key} {case['id']} {status:8s} "
                f"hit={records[key][-1]['source_hit']!s:5s} {finished - started:5.1f}s"
            )

    pipeline_logger.removeHandler(recorder)

    output = {
        "run": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "generator": f"{LLM_PROVIDER}/{LLM_MODEL}",
            "generator_temperature": TEMPERATURE,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "indexed_chunks": get_collection().count(),
            "golden_cases": len(golden),
            "repo_head": _git("rev-parse", "--short", "HEAD"),
            "corpus_commit": _git("log", "-1", "--format=%h", "--", "data/standardized", "chroma_db"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
        },
        "configs": {key: {**config, "records": records[key]} for key, config in CONFIGS.items()},
    }
    _write_json(GENERATIONS_PATH, output)
    print(f"Saved {GENERATIONS_PATH.relative_to(ROOT)}")
    return output


# ------------------------------------------------------------------ score


def _build_metrics(names: list[str]) -> dict:
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY cho evaluator")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    llm = llm_factory(
        EVAL_LLM_MODEL,
        provider="openai",
        client=AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL),
        adapter="instructor",
        cache=DiskCacheBackend(cache_dir=str(CACHE_DIR)),
        temperature=0,
    )
    builders = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=_evaluator_embeddings(api_key)),
        "context_recall": lambda: ContextRecall(llm=llm),
        "context_precision": lambda: ContextPrecisionWithReference(llm=llm),
    }
    return {name: builders[name]() for name in names}


def _evaluator_embeddings(api_key: str):
    """Embedding cho answer relevancy, gọi từng text một.

    gemini-embedding-2 gộp cả list đầu vào thành một vector duy nhất, nên không
    được gửi batch như gemini-embedding-001.
    """
    from google import genai
    from ragas.embeddings import GoogleEmbeddings

    class PerTextGoogleEmbeddings(GoogleEmbeddings):
        def embed_texts(self, texts, **kwargs):
            return [self.embed_text(text, **kwargs) for text in texts]

    return PerTextGoogleEmbeddings(client=genai.Client(api_key=api_key), model=EVAL_EMBEDDING_MODEL)


def _metric_kwargs(metric: str, record: dict) -> dict:
    if metric == "faithfulness":
        return {"user_input": record["question"], "response": record["answer"], "retrieved_contexts": record["contexts"]}
    if metric == "answer_relevancy":
        return {"user_input": record["question"], "response": record["answer"]}
    if metric == "context_recall":
        return {"user_input": record["question"], "retrieved_contexts": record["contexts"], "reference": record["expected_answer"]}
    return {"user_input": record["question"], "reference": record["expected_answer"], "retrieved_contexts": record["contexts"]}


def _as_float(value) -> float | None:
    value = getattr(value, "value", value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else round(number, 4)


RETRY_AFTER = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


class DailyQuotaExhausted(RuntimeError):
    """Hết quota theo ngày: retry vô ích, dừng để chạy lại sau (cache giữ phần đã chấm)."""


def _error_text(error: BaseException) -> str:
    """Gom thông điệp của cả chuỗi exception.

    instructor bọc lỗi HTTP gốc trong InstructorRetryException, nên quotaId
    ("...PerDay...") chỉ nằm ở lỗi gốc hoặc trong danh sách lần thử thất bại.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts += [str(current), repr(current)]
        for attribute in ("failed_attempts", "last_completion"):
            value = getattr(current, attribute, None)
            if value:
                parts.append(repr(value))
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def _retry_delay(error: Exception, attempt: int) -> float | None:
    """Số giây chờ trước khi thử lại, None nếu lỗi không nên retry."""
    text = _error_text(error)
    if "PerDay" in text:
        raise DailyQuotaExhausted(text[:300])
    lowered = text.lower()
    if not any(marker in lowered for marker in ("429", "503", "resource_exhausted", "unavailable", "overloaded")):
        return None
    # Free tier trả sẵn thời điểm reset; chờ đúng mức đó thay vì đoán.
    match = RETRY_AFTER.search(text)
    return float(match.group(1)) + 1 if match else 5 * 2**attempt


async def _score_one(semaphore, scorer, metric: str, record: dict, label: str) -> float | None:
    if not record["contexts"] and metric != "answer_relevancy":
        return 0.0
    async with semaphore:
        for attempt in range(METRIC_MAX_ATTEMPTS):
            try:
                return _as_float(await scorer.ascore(**_metric_kwargs(metric, record)))
            except DailyQuotaExhausted:
                raise
            except Exception as error:
                delay = _retry_delay(error, attempt)
                if attempt == METRIC_MAX_ATTEMPTS - 1 or delay is None:
                    print(f"  {label} {metric}: lỗi, ghi None — {type(error).__name__}: {str(error)[:160]}")
                    return None
                await asyncio.sleep(delay)
    return None


async def _score_all(generations: dict, names: list[str], existing: dict) -> dict:
    """Chấm các ô (config, case, metric) còn thiếu; giữ nguyên ô đã có điểm."""
    scores = {key: {case: dict(values) for case, values in existing.get(key, {}).items()}
              for key in generations["configs"]}
    jobs = []
    for key, config in generations["configs"].items():
        for record in config["records"]:
            case_scores = scores[key].setdefault(record["id"], {})
            for metric in names:
                if case_scores.get(metric) is None:
                    jobs.append((key, record, metric))

    if not jobs:
        print("Không còn ô nào cần chấm.")
        return scores

    print(f"Chấm {len(jobs)} ô bằng {EVAL_LLM_MODEL}...")
    metrics = _build_metrics(sorted({metric for _, _, metric in jobs}))
    semaphore = asyncio.Semaphore(METRIC_CONCURRENCY)
    values = await asyncio.gather(
        *(_score_one(semaphore, metrics[metric], metric, record, f"{key} {record['id']}")
          for key, record, metric in jobs),
        return_exceptions=True,
    )

    quota_errors = 0
    for (key, record, metric), value in zip(jobs, values):
        if isinstance(value, BaseException):
            quota_errors += isinstance(value, DailyQuotaExhausted)
            value = None
        scores[key][record["id"]][metric] = value
    if quota_errors:
        print(f"{quota_errors} ô chưa chấm vì hết quota ngày; chạy lại lệnh score sau khi quota reset.")
    return scores


def _mean(values: list[float | None]) -> tuple[float | None, int]:
    valid = [value for value in values if value is not None]
    return (round(statistics.mean(valid), 4) if valid else None, len(valid))


def summarize(generations: dict, scores: dict) -> dict:
    summary = {"run": generations["run"], "evaluator": EVAL_LLM_MODEL, "configs": {}}
    for key, config in generations["configs"].items():
        records = config["records"]
        per_case = []
        for record in records:
            case_scores = scores[key][record["id"]]
            case_mean, _ = _mean([case_scores.get(metric) for metric in METRICS])
            per_case.append({"id": record["id"], "retrieval_type": record["retrieval_type"],
                             "status": record["status"], "source_hit": record["source_hit"],
                             **case_scores, "mean": case_mean})

        metric_means = {}
        for metric in METRICS:
            mean, count = _mean([scores[key][record["id"]].get(metric) for record in records])
            metric_means[metric] = {"mean": mean, "n": count}
        average, _ = _mean([value["mean"] for value in metric_means.values()])

        by_type: dict[str, list[float | None]] = {}
        for row in per_case:
            by_type.setdefault(row["retrieval_type"], []).append(row["mean"])

        latency = {
            field: {
                "median": round(statistics.median(record[field] for record in records), 3),
                "p90": round(sorted(record[field] for record in records)[int(0.9 * (len(records) - 1))], 3),
            }
            for field in ("latency_retrieval_s", "latency_generation_s", "latency_total_s")
        }
        statuses: dict[str, int] = {}
        for record in records:
            statuses[record["status"]] = statuses.get(record["status"], 0) + 1

        summary["configs"][key] = {
            "name": config["name"],
            "metrics": metric_means,
            "average": average,
            "by_retrieval_type": {name: _mean(values)[0] for name, values in by_type.items()},
            "source_hit_rate": round(sum(record["source_hit"] for record in records) / len(records), 4),
            "fallback_attempted": sum(record["fallback_attempted"] for record in records),
            "statuses": statuses,
            "latency": latency,
            "mean_context_chars": round(statistics.mean(record["context_chars"] for record in records)),
            "per_case": per_case,
        }

    delta = {}
    for metric in METRICS:
        a = summary["configs"]["A"]["metrics"][metric]["mean"]
        b = summary["configs"]["B"]["metrics"][metric]["mean"]
        delta[metric] = round(b - a, 4) if a is not None and b is not None else None
    a_avg, b_avg = summary["configs"]["A"]["average"], summary["configs"]["B"]["average"]
    delta["average"] = round(b_avg - a_avg, 4) if a_avg is not None and b_avg is not None else None
    summary["delta_b_minus_a"] = delta
    return summary


def run_scoring(names: list[str] | None = None) -> dict:
    generations = _read_json(GENERATIONS_PATH)
    existing = _read_json(SCORES_PATH) if SCORES_PATH.exists() else {}
    started = time.perf_counter()
    scores = asyncio.run(_score_all(generations, names or list(METRICS), existing))
    _write_json(SCORES_PATH, scores)
    summary = summarize(generations, scores)
    summary["scoring_seconds"] = round(time.perf_counter() - started, 1)
    _write_json(SUMMARY_PATH, summary)
    print(f"Saved {SCORES_PATH.relative_to(ROOT)} and {SUMMARY_PATH.relative_to(ROOT)}")
    print_report(summary)
    return summary


def print_report(summary: dict) -> None:
    a, b = summary["configs"]["A"], summary["configs"]["B"]
    print(f"\n{'metric':20s} {'A':>8s} {'B':>8s} {'B-A':>8s}")
    for metric in METRICS:
        print(f"{metric:20s} {a['metrics'][metric]['mean']!s:>8} {b['metrics'][metric]['mean']!s:>8} "
              f"{summary['delta_b_minus_a'][metric]!s:>8}   (n={a['metrics'][metric]['n']}/{b['metrics'][metric]['n']})")
    print(f"{'average':20s} {a['average']!s:>8} {b['average']!s:>8} {summary['delta_b_minus_a']['average']!s:>8}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    # Tuỳ chọn: chỉ chấm một số metric, ví dụ `score faithfulness,context_recall`.
    selected = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    if selected and not set(selected) <= set(METRICS):
        raise SystemExit(f"Metric không hợp lệ: {selected}; chọn trong {METRICS}")
    if stage in ("generate", "all"):
        run_generation()
    if stage in ("score", "all"):
        run_scoring(selected)
