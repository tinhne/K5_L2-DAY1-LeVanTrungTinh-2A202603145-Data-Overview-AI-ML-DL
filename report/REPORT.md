# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11

**Ngày chạy:** 6/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Ultralytics `8.4.145`;

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** không thay đổi

## 1. Phân loại ảnh

Nguồn: `classification_predictions.json`, record có `sample_id = traffic` và `rank = 1`.

1. Record hạng 1:

   `class_id=468`; `class_name=cab`; `rank=1`; `score=0.510915`; `taxonomy_name=ImageNet-1K`.

2. Đây là dự đoán cho **cả bức ảnh**, không phải tọa độ của một vật thể riêng, vì record classification không có `bbox` hoặc polygon; nó chỉ có một lớp và điểm cho mỗi ảnh.
3. Danh sách tên lớp đi kèm taxonomy/checkpoint ImageNet-1K, không phải model tự tạo ra tùy ý.
4. `class_id` là mã ổn định, `class_name` dễ đọc, còn `taxonomy_name` cho biết lớp đó thuộc bộ nhãn nào. Ghi cả ba giúp đối chiếu, tái lập và tránh nhầm tên lớp giữa các taxonomy.
5. Guideline cần quy định chủ thể/chủ đề chính, cách xử lý ảnh có nhiều vật, và nguyên tắc chọn đúng một lớp khi các vật cùng xuất hiện.
6. `score` chỉ là mức tin cậy của model, không phải ground truth. Ground truth phải do annotator gán theo guideline và được reviewer kiểm tra.

## 2. Phát hiện vật thể

Nguồn: `detection_predictions.json`, các record có `sample_id = kitchen`. Chọn record có điểm cao nhất:

1. `class_name=person`; `score=0.912625`; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`; `bbox_width=113.58`; `bbox_height=279.68`.
2. Với ảnh rộng 640 và cao 427 pixel, box bắt đầu ở góc trên-trái khoảng `(385.33, 69.24)` và kết thúc ở góc dưới-phải khoảng `(498.92, 348.92)`. Box rộng 113.58 pixel và cao 279.68 pixel.
3. JSON hiện chỉ lưu kết quả được tạo ở ngưỡng `0.35`: ảnh `kitchen` có **11 prediction**. Không có output so sánh ở ngưỡng `0.20` và `0.60`, nên không thể xác định chính xác hai số đó từ file này.
4. Khi hạ ngưỡng, reviewer phải xem thêm nhiều box hơn, loại false positive, kiểm tra box trùng/lỗi và xác nhận các vật có điểm thấp.
5. Quy tắc box: box phải bao phủ toàn bộ phần nhìn thấy của đúng một vật, sát biên vật thể nhất có thể, không kéo sang vật khác; vật bị cắt bởi mép ảnh thì box được phép chạm mép ảnh.
6. Trên ảnh `kitchen`, có vật bị cắt ở mép trái: một prediction `person` có `x1=0.12`, và prediction `oven` có `x1=0.11`. Đây là tình huống nên quy định trong guideline (box bám phần nhìn thấy và chạm mép ảnh); nếu guideline chưa có quy tắc cho vật bị cắt thì cần escalation để thống nhất.

## 3. Phân đoạn theo từng đối tượng

Nguồn: `segmentation_predictions.json`, chọn record `kitchen-001`.

1. `instance_id=kitchen-001`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; 5 điểm đầu của `polygon_xy` là `[[446,70], [445,71], [444,71], [443,72], [442,72]]`.
2. Box chỉ mô tả hình chữ nhật bao quanh; polygon cho biết biên dạng thực tế của từng vật và phần diện tích thuộc về vật.
3. `instance_id` định danh một đối tượng cụ thể trong ảnh, kể cả khi có nhiều đối tượng cùng class. Nó không phải `class_id` (mã loại vật) và không phải tracking ID (định danh xuyên suốt nhiều frame).
4. Quy tắc mask: vẽ polygon theo biên phần nhìn thấy của đúng vật, bám sát biên nhưng không ăn sang nền/vật bên cạnh; phần bị che khuất không được tự đoán nếu guideline không cho phép.
5. Một trường hợp khó là `kitchen-004` (`potted plant`) có polygon bắt đầu gần mép trái và mép trên (`x=1`, `y=2`), cho thấy vật bị cắt bởi biên ảnh. Trường hợp này cần áp dụng quy tắc vật bị cắt trong guideline; nếu chưa có quy tắc thì escalation.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Ground truth trông như thế nào?                                    | Lỗi hoặc chỗ khó quyết định thấy trong output                                                 | Annotator làm gì?                                                    | Reviewer xem gì?                                                              |
| --------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Phân loại ảnh         | Một `class_id/class_name` cho toàn ảnh, theo taxonomy đã quy định. | Ảnh `traffic` có top-1 `cab` với score chỉ `0.510915`; model score không phải nhãn đúng.      | Chọn một lớp theo chủ thể/chủ đề chính và guideline.                 | Kiểm tra lớp có đúng taxonomy, đúng chủ đề chính và nhất quán không.          |
| Phát hiện vật thể     | Một hoặc nhiều box `xyxy` và class cho từng vật.                   | `kitchen` có 11 prediction ở ngưỡng 0.35; có box chạm mép ảnh và nhiều vật nhỏ có score thấp. | Vẽ box cho từng instance nhìn thấy, xử lý vật bị cắt theo guideline. | Kiểm tra class, số box, biên box, box trùng và false positive.                |
| Instance segmentation | Một polygon mask cho từng instance, kèm class.                     | `kitchen-004` có polygon 100 điểm và chạm gần mép ảnh; biên bị che/cắt cần quy tắc rõ.        | Vẽ mask bám biên phần nhìn thấy của từng vật.                        | Kiểm tra polygon kín, bám đúng biên, không lấn nền/vật khác và đúng instance. |

## 5. An toàn dữ liệu

1. Chỉ sử dụng các file output cần cho bài lab, không thêm hoặc công khai thông tin cá nhân hay dữ liệu ngoài phạm vi.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, dừng xử lý và báo Lab Coach/reviewer để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md`
- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `classification_top5.png`
- [x] `detection_predictions.png`
- [x] `segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS` 
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp link repository trên LMS
