# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Nguyễn Đức Tâm
- Mã học viên: 2A202602921
- Nhóm: THTrueMi
- Repository/branch:https://github.com/tamnd2004/K4-L3B-RAG-Pipeline-THTrueMi

## Phần việc đã thực hiện

| Module/deliverable               | Việc tôi trực tiếp làm                                                                                                                                                                                                                                                          | File/commit/PR                                                                                                                        | Trạng thái |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Thu thập dữ liệu (Task 1–2)      | Tải 3 PDF playbook large migration (kiểm tra `%PDF`, chạy lại không tải trùng); crawl 5 trang AWS Prescriptive Guidance bằng Crawl4AI với `target_elements=#main-col-body` để bỏ nav mà vẫn giữ `<title>` thật; bỏ tham số `utm_source` khỏi URL để citation trỏ đúng trang gốc | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `data/landing/` · commit `7085182`                                      | Done       |
| Chunk, embed, index (Task 4)     | Đọc header của Task 3 để lấy title/URL thật; thêm metadata `section`; chunk recursive 500/50; embed `gemini-embedding-001` (3072 chiều) có retry khi gặp 429; upsert theo ID ổn định và xoá chunk mồ côi; tự tạo lại collection khi đổi dimension                               | `src/task4_chunking_indexing.py`, `chroma_db/` · commit `ca3bb0a`                                                                     | Done       |
| Dense + BM25 (Task 5–6)          | `semantic_search` dùng lại đúng `embed_texts` của Task 4; BM25 tokenize bằng regex, lọc theo token overlap, cache index và token set                                                                                                                                            | `src/task5_semantic_search.py`, `src/task6_lexical_search.py` · commit `ca3bb0a`                                                      | Done       |
| Generation có citation (Task 10) | Trên bản của huyhoangg1706 (`263a4b1`): sửa lỗi citation lệch sau reorder, kiểm tra citation, trả safe refusal khi không còn citation hợp lệ, retry 429/503, cố định ngôn ngữ trả lời, đổi generator sang `gemini-3.5-flash-lite`                                               | `src/task10_generation.py` · chưa commit                                                                                              | Done       |
| Chatbot UI                       | `app.py`: gọi `generate_with_citation(query, top_k)`; hiển thị answer, nguồn (title, mục, URL lấy từ metadata, method, score), đánh dấu nguồn được trích; lưu đủ `sources` vào `session_state` để render lại lịch sử                                                            | `app.py` · chưa commit                                                                                                                | Done       |
| Evaluation A/B                   | Script 2 giai đoạn (generate/score) chạy 15 case × 2 config, 4 metric RAGAS; phân tích 3 case kém nhất; viết `RESULT.md`                                                                                                                                                        | `group_project/evaluation/run_evaluation.py`, `group_project/evaluation/results/`, `group_project/evaluation/RESULT.md` · chưa commit | Done       |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Đánh số citation `[n]` theo thứ hạng trong `sources` _trước_ khi reorder, và coi câu trả lời không còn citation hợp lệ nào là safe refusal.  
   **Lý do/evidence:** Bản trước gán nhãn theo vị trí sau reorder. Tôi tái hiện được 3/5 nhãn trỏ sai chunk: model trích `[Document 2]` (nội dung chunk-2) nhưng UI hiển thị `sources[1]` là chunk-1. Sau khi sửa còn 0/5 nhãn lệch, và AppTest xác nhận mọi `[n]` trong câu trả lời đều map đúng nguồn đang hiển thị.  
   **Trade-off:** Context đưa cho LLM có nhãn không liên tục (`[1] [3] [5] [4] [2]`). Luật "không citation thì từ chối" an toàn nhưng có thể từ chối một câu trả lời đúng mà model quên trích; trên golden set tỷ lệ này là 0/30.

2. **Quyết định:** Embedding qua Gemini API (`gemini-embedding-001`) thay vì tải `BAAI/bge-m3` về máy.  
   **Lý do/evidence:** Không phải tải khoảng 2.2 GB, mà vẫn tìm được xuyên ngôn ngữ: query tiếng Việt về cutover lấy đúng đoạn tiếng Anh với cosine 0.75–0.77.  
   **Trade-off:** Free tier giới hạn 1000 request embedding/ngày. Index 951 chunk tiêu gần hết quota, nên trong cùng ngày dense search ngừng hoạt động (pipeline lùi về BM25, không crash); rate limit cũng làm lần index mất khoảng 5 phút.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest tests/test_contracts.py` (15/15 pass) và `pytest tests/test_acceptance.py`; query tiếng Việt/tiếng Anh trong domain và ngoài domain ("Công thức nấu phở bò là gì?", "What is the capital of France?"); `streamlit.testing.AppTest` cho UI; 15 golden case × 2 config cho evaluation.
- Kết quả trước/sau nếu có: citation lệch 3/5 → 0/5; câu hỏi tiếng Anh bị trả lời bằng tiếng Việt → 2/2 đúng ngôn ngữ; câu hỏi ngoài domain → safe refusal với `sources=[]`, `retrieval_source="none"`, UI không crash. A/B trên 15 golden case: average A 0.7852, B 0.7853; B hơn ở faithfulness (+0.0711, nhưng phần lớn là nhiễu của evaluator ở case_07/14) và thua ở context precision (−0.0841); chi tiết trong `group_project/evaluation/RESULT.md`.
- Lỗi đã phát hiện và cách xử lý:
  - BM25 trả list rỗng vì `BM25Okapi` cho idf = 0 trên corpus nhỏ → lọc theo token overlap thay vì `score > 0`.
  - Console Windows (cp1252) crash khi in tiếng Việt → chuyển stdout sang UTF-8 trong `src/__init__.py`.
  - `.env` trỏ tới `gemini-2.5-flash-lite`, model đã bị khoá với tài khoản mới, nên mọi câu trả lời thành safe refusal → đổi sang `gemini-3.5-flash-lite`.
  - Evaluator `gemini-3.6-flash` chỉ có 20 request/ngày ở free tier → đổi sang `gemini-3.1-flash-lite` và cho script dừng sớm khi hết quota ngày.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: 628/628 chunk từ PDF có `section` trùng tên tài liệu vì bước convert PDF làm mất heading, nên chunk 500 ký tự cắt ngang ranh giới chủ đề. Hệ quả thấy rõ ở case_02: cả hai config đều trượt đoạn định nghĩa platform/people foundation.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: khi retrieve, mở rộng sang chunk lân cận cùng `section` (small-to-big) để lấy trọn danh sách bị cắt như ở case_15, rồi chạy lại `run_evaluation.py` để đo context recall trước/sau.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Nguyễn Đức Tâm
