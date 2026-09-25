# RAG evaluation results

Chatbot tra cứu kiến thức migration lên AWS: so sánh Config A (dense-only) và Config B (hybrid + RRF) trên cùng golden dataset.

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 (generate 12:39, score 13:00, giờ Việt Nam) |
| Framework and version              | RAGAS 0.4.3 (`ragas.metrics.collections`: `Faithfulness`, `AnswerRelevancy` strictness=3, `ContextRecall`, `ContextPrecisionWithReference`), Python 3.12.8 |
| Evaluator model                    | `gemini-3.1-flash-lite`, temperature 0; embedding cho answer relevancy: `gemini-embedding-2` |
| Generator model                    | `gemini-3.5-flash-lite`, temperature 0.3, top_p 0.9, max 1024 token; cùng `SYSTEM_PROMPT` và hàm `answer_with_sources` cho cả hai config |
| Embedding model                    | `gemini-embedding-001` (3072 chiều), ChromaDB cosine |
| Corpus version/commit              | Corpus và index tại commit `ca3bb0a` (3 playbook PDF + 5 trang web, 951 chunk, recursive 500/50). Code chạy từ HEAD `448f456` cộng các thay đổi chưa commit ở `src/task10_generation.py` và `group_project/evaluation/` |
| Golden dataset size                | 15 case: 5 keyword, 5 semantic, 5 source_disambiguation |
| `top_k`                            | 5 chunk đưa vào context; mỗi nhánh retrieval lấy 10 ứng viên |
| Fallback threshold and calibration | Cosine 0.65, do nhóm hiệu chỉnh ở Task 9 (in-domain 0.714–0.880, out-of-domain 0.505–0.574). Trong lần chạy này, best cosine của golden set là 0.726–0.884, nên 0/30 lượt kích hoạt PageIndex fallback |

Chạy lại: `python group_project/evaluation/run_evaluation.py generate` rồi `... score`. Dữ liệu thô nằm trong `results/generations.json` (answer, contexts, latency), `results/scores.json` và `results/summary.json`.

## Configurations

- **Config A — dense-only:** `retrieve(query, top_k=5, use_reranking=False)`: lấy 10 chunk gần nhất theo cosine trong ChromaDB rồi giữ 5 chunk đầu.
- **Config B — hybrid + RRF:** `retrieve(query, top_k=5, use_reranking=True)`: dense top-10 và BM25 top-10, fuse một lần bằng RRF (k=60), giữ 5 chunk đầu.

Hai config dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ khác retrieval strategy. A và B chạy xen kẽ theo từng câu để giới hạn rate limit ảnh hưởng đều lên cả hai.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |   0.8600 |   0.9311 |   +0.0711 |
| Answer relevance  |   0.8722 |   0.8854 |   +0.0132 |
| Context recall    |   0.7556 |   0.7556 |    0.0000 |
| Context precision |   0.6530 |   0.5689 |   −0.0841 |
| **Average**       | **0.7852** | **0.7853** | **+0.0001** |

Theo loại câu hỏi (trung bình 4 metric):

| Loại câu hỏi          | Config A | Config B | Delta B−A |
| --------------------- | -------: | -------: | --------: |
| keyword               |   0.8569 |   0.7718 |   −0.0851 |
| semantic              |   0.6855 |   0.7589 |   +0.0734 |
| source_disambiguation |   0.8132 |   0.8250 |   +0.0118 |

Cả hai config trả lời 15/15 câu (không có refusal), và 15/15 lần lấy được đúng tài liệu nguồn của golden case.

## A/B comparison

- **Cấu hình tốt hơn:** Không có cấu hình thắng rõ ràng. Average chênh +0.0001, và hai chênh lệch lớn nhất đi ngược chiều nhau: B tăng faithfulness nhưng giảm context precision. Pipeline sản phẩm vẫn chạy B vì contract của Task 9 yêu cầu hybrid + RRF, nhưng lần đo này **không chứng minh được RRF là một cải tiến** so với dense-only.
- **Evidence:**
  - *Faithfulness +0.0711 chủ yếu là nhiễu của evaluator.* Ở case_07, câu trả lời của A và B gần như giống từng chữ, cùng trích một chunk và hai config chung 4/5 chunk, nhưng bị chấm 0.40 và 1.00. Ở case_14, hai câu trả lời có cùng nội dung, bị chấm 0.25 và 0.67. Bỏ hai case này thì chênh lệch còn +0.0038 (A 0.9423, B 0.9462).
  - *Context recall bằng nhau nhưng khác case.* B lấy thêm được bằng chứng ở hai câu semantic, case_02 (0.5 → 1.0) và case_03 (0.0 → 0.5), nhưng mất trọn bằng chứng ở case_09 (1.0 → 0.0).
  - *Context precision −0.0841 là thật và lặp lại:* BM25 thêm chunk cùng tài liệu nhưng sai mục (case_07 1.00 → 0.33, case_11 1.00 → 0.50, case_09 0.70 → 0.33). Nghịch lý là B tệ nhất ở nhóm keyword (−0.0851): câu hỏi keyword trong corpus này dùng tên dịch vụ ("PostgreSQL", "Aurora", "Application Migration Service") lặp lại ở gần như mọi chunk của cùng một bài, nên BM25 không phân biệt được mục nào chứa đáp án.
  - Với 15 case, chạy một lần, generator temperature 0.3 và evaluator dao động tới 0.6 điểm trên cùng một câu trả lời, mọi chênh lệch dưới khoảng 0.05 nên coi là nhiễu.
- **Trade-off về latency/cost:**
  - *Latency:* median retrieval 0.384s (A) so với 0.391s (B), tức BM25 chạy local chỉ thêm khoảng 7 ms. Median tổng 2.44s so với 2.36s, chủ yếu là thời gian generation, chênh lệch nằm trong nhiễu mạng. p90 tổng của A (3.87s) cao hơn B (2.82s) chỉ vì hai lượt A phải chờ quota (case_09 19.3s, case_15 78.0s do retry 429), không phải do chi phí của pipeline.
  - *Cost:* mỗi query tốn đúng 1 embedding và 1 generation ở cả hai config. Context của B dài hơn khoảng 3% (2047 so với 1979 ký tự), tương ứng khoảng +3% input token.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | case_03 — "A team wants to move as many applications as possible in each batch. What should constrain the size and timing of those batches?" | A | 1.0000 | 0.7025 | 0.0000 | 0.3333 | retrieval (chunking) | Danh sách tiêu chí bắt đầu ở chunk-5 của portfolio playbook (capacity của migration workstream) và tiếp tục ở chunk-6 (dependencies, budgets, performance goals, resource availability, deadlines). Chunk-6 không được lấy ở cả hai config, nên câu trả lời chỉ nêu capacity. Chunk từ PDF không có heading (628/628 chunk có `section` trùng tên tài liệu), nên nhát cắt 500 ký tự rơi vào giữa danh sách. |
|   2 | case_09 — "Which migration options are described for moving an on-premises PostgreSQL database to Aurora PostgreSQL?" | B | 1.0000 | 0.9601 | 0.0000 | 0.3333 | retrieval (fusion) | Config A có chunk-13 và chunk-3 (native tools pg_dump, pg_restore, psql, third-party tools), recall 1.0. Ở B, BM25 xếp chunk-5 (Prerequisites) và chunk-53 (Migrate the application) lên cao vì chúng lặp lại "PostgreSQL/Aurora/migration". RRF chia trọng số đều cho hai nhánh nên đẩy chunk-13 và chunk-3 ra khỏi top-5, và câu trả lời của B mất phần native tools. |
|   3 | case_15 — "According to the Cutover stage guidance, what data-handling options should be considered when rolling back after the migrated application has received new transactions?" | A (B giống hệt: 0.5 / 0.0) | 1.0000 | 0.8490 | 0.5000 | 0.0000 | retrieval (chunking) | Cả hai config lấy được chunk-27, là câu dẫn "We recommend that you consider the following a…" (bị cắt ngay giữa chữ), nhưng trượt chunk-28/29 chứa ba phương án: fail-forward bằng AWS DMS, dual write, native backup/restore. Câu trả lời chỉ nói "consider the approaches" mà không nêu được phương án nào. |

Case #1 và #3 có chung một kiểu lỗi với case_01 và case_10: **phần còn lại của danh sách nằm ở chunk kế tiếp và không được retrieve.** Ở case_01, 2/4 core workstream nằm ở chunk-15 (lấy được), 2/4 còn lại ở chunk-16 (trượt), nên model trả lời đúng rằng context chỉ nêu hai. Ở case_10, CPU và IOPS nằm ở chunk-18/19, còn memory footprint ở chunk-20 (trượt). Recall của case_10 bằng 0 dù đã có 2/3 ý, vì reference là một câu duy nhất và RAGAS chấm đúng/sai theo cả câu.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
| 1 | Mở rộng chunk lân cận (small-to-big): với mỗi chunk trong top-k, thêm chunk `chunk_index + 1` cùng `source` (với trang web thì cùng `section`), bỏ trùng, giới hạn tổng context. Không phải index lại nên không tốn quota embedding. | 4/15 case (01, 03, 10, 15), ở cả hai config, trượt đúng chunk kế tiếp chứa phần còn lại của danh sách: chunk-16, chunk-6, chunk-20, chunk-28/29. | Recall của case_03, case_10, case_15 lên ≥ 0.67; case_01 nêu đủ 4 workstream. Context dài hơn khoảng 1.5–2 lần, nên precision có thể giảm và input token tăng tương ứng. | Chạy lại `run_evaluation.py generate` + `score` cho cả A và B. Kiểm tra (a) chunk-16/6/20/28 có trong `retrieved` của các case tương ứng, (b) context recall tổng và precision so với bảng trên, (c) `mean_context_chars` và latency trong `summary.json`. |
| 2 | Hạ trọng số BM25 trong RRF (ví dụ dense 1.0, BM25 0.5), hoặc chỉ cho BM25 tham gia khi query có token hiếm (idf cao) như mã dịch vụ hay tên lệnh. | case_09: B mất trọn bằng chứng (recall 1.0 → 0.0) vì BM25 ưu tiên chunk lặp tên dịch vụ. Precision của B thấp hơn A 0.0841, và thấp hơn 0.0851 ở nhóm keyword. | Lấy lại recall 1.0 cho case_09 và đưa precision của B về mức ≥ A (0.6530), trong khi giữ phần B đang hơn ở case_02/03 (semantic +0.0734). | Chạy lại config B trên 15 case. So sánh recall của case_09, precision tổng, điểm theo loại câu hỏi và recall của case_02/03 với bảng trên. Thêm một test contract cho `rerank_rrf` có trọng số. |
| 3 | Tăng độ tin cậy của phép đo: tách `expected_answer` thành từng ý một câu, chạy mỗi config 3 lần và báo mean ± độ lệch, dùng evaluator mạnh hơn (`gemini-3.6-flash` chỉ cho 20 request/ngày ở free tier nên cần paid tier hoặc chia ra nhiều ngày). | Evaluator chấm lệch 0.60 (case_07) và 0.42 (case_14) trên hai câu trả lời gần như giống hệt nhau, tạo ra phần lớn chênh lệch faithfulness +0.0711. case_10 bị recall 0 dù context có 2/3 ý, vì reference chỉ là một câu. | Tách được nhiễu của evaluator khỏi hiệu ứng thật của retrieval; recall phản ánh độ phủ từng phần; kết luận A/B có khoảng tin cậy. | Chấm lại case_07 và case_14 bằng evaluator mới: điểm faithfulness của A và B phải hội tụ (chênh < 0.1). Sau khi tách ý, recall của case_10 phải ≈ 0.67. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Không chạy bonus experiment trong lần đánh giá này | — | — | — | Lần đánh giá chỉ gồm A/B bắt buộc. Recommendation 1 và 2 ở trên là hai thí nghiệm nên chạy tiếp, dùng Config A/B hiện tại làm baseline. |

## Giới hạn của lần đánh giá

- Golden set gồm 15 câu hỏi **tiếng Anh**, trong khi người dùng thật hỏi bằng tiếng Việt. Khi thử thủ công, câu hỏi ghép tiếng Việt về cutover không lấy được phần "cần kiểm tra gì trước cutover", nhưng loại lỗi này chưa được đo.
- Mỗi config chỉ chạy một lần. Generator (temperature 0.3) và evaluator lite đều dao động, nên chênh lệch dưới khoảng 0.05 không đủ để kết luận.
- PageIndex chưa được khởi tạo trên máy chạy đánh giá, nên nhánh fallback không được đo; trong lần chạy này cũng không câu nào xuống dưới threshold.
