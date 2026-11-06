# Hướng dẫn học viên – Ngày 1

## Trước khi chạy

Đây là bài thực hành quan sát bằng Google Colab. Không cần cài CVAT hoặc Python trên máy cá nhân. Lần đầu
chạy sẽ cài đúng phiên bản Ultralytics, tải ba checkpoint YOLO11n và ba ảnh mẫu; hãy chờ ô mã hoàn tất.
Luồng chuẩn giống nhau trên Windows, macOS và Ubuntu vì mọi thao tác bắt buộc đều chạy trong trình duyệt;
Git/terminal chỉ là lựa chọn thêm cho học viên đã quen.

**Bài nộp = link repository** có `report/REPORT.md` và `report/day1_lab_outputs/`. Không nộp ZIP Drive.

Preflight:

- [ ] Đã mở notebook từ repository nguồn bằng Colab và nhấn **Copy to Drive**.
- [ ] Biết điền `REPORT.md` từ `reports/REPORT_TEMPLATE.md` trên máy, rồi dùng ô Upload trong notebook.
- [ ] Không đưa ảnh hoặc dữ liệu nhạy cảm vào Colab; tên repository theo mẫu `<KHOA>-L2-DAY01-HoVaTen-MSSV-Data-Overview-AI-ML-DL`.
- [ ] Đã tạo repository bài làm từ template và biết thư mục nộp bài là `report/`.
- [ ] Biết cách upload thư mục lên GitHub Web (hoặc dùng Git nếu đã quen).

## Quy trình 60 phút

Mốc thời gian dưới đây là hướng dẫn vận hành, chưa phải ngưỡng chấm chính thức:

| Phút | Việc cần làm | Evidence |
| ---: | --- | --- |
| 0–5 | Mở notebook, đọc scope và chạy setup | phiên bản + device |
| 5–15 | Chạy classification, đọc top-5 | JSON + biểu đồ |
| 15–30 | Chạy detection, đọc một box và so sánh threshold | JSON + ảnh box |
| 30–45 | Chạy instance segmentation, đọc một polygon | JSON + ảnh mask |
| 45–55 | Điền `REPORT.md` trên máy; chạy ô Upload trong Colab | báo cáo |
| 55–60 | Chạy ô đóng ZIP → giải nén vào `report/` trên GitHub → nộp **link repo** | link repo |

## Cách đọc đầu ra

| Tệp | Một record là gì? | Trường cần đọc |
| --- | --- | --- |
| `classification_predictions.json` | một lớp được xếp hạng cho toàn ảnh | `class_id`, `class_name`, `rank`, `score`, `taxonomy_name` |
| `detection_predictions.json` | một vật thể được model phát hiện | `class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height` |
| `segmentation_predictions.json` | một instance và đa giác của nó | `instance_id`, `class_name`, `score`, `polygon_xy` |

`classification_top5.png` minh họa sample `traffic`; hai hình detection/segmentation minh họa sample
`kitchen`. Khi trích JSON, luôn ghi `sample_id` để record khớp đúng hình. Sample `dining` cũng có trong JSON
để đủ ba ảnh; báo cáo dùng `traffic` (phân loại) và `kitchen` (detection/segmentation).

### Thuật ngữ nhanh

- **Taxonomy / checkpoint:** danh sách lớp cố định đi kèm model `.pt`.
- **Guideline:** quy tắc gán nhãn do người viết, áp dụng được nhiều ảnh.
- **Annotator / Reviewer:** người gán nhãn / người kiểm tra.
- **Escalation:** ca khó hoặc rule chưa phủ → hỏi Lab Coach, không tự bịa nhãn.
- **Model score:** dùng xếp hạng/lọc prediction; không phải điểm chất lượng ground truth.

### Phân loại ảnh

- Checkpoint `yolo11n-cls.pt` xếp hạng các lớp ImageNet-1K cho toàn ảnh.
- `rank=1` là lớp có model score cao nhất, không phải ground truth đã được con người xác nhận.
- Ảnh COCO + taxonomy ImageNet có thể cho top-1 trông “lạ” — đó là tư liệu học.
- “Một lớp cho mỗi ảnh” chỉ là ví dụ single-label; dự án khác có thể multi-label.

### Phát hiện vật thể

- `bbox_xyxy = [x_min, y_min, x_max, y_max]` dùng pixel, gốc tọa độ ở góc trên trái.
- Hạ threshold thường giữ thêm prediction; nâng threshold thường loại thêm prediction.
- Threshold là cấu hình lọc prediction, không phải quy tắc bỏ qua object khi tạo ground truth.

### Phân đoạn theo từng đối tượng

- `polygon_xy` là danh sách các điểm `[x, y]` theo pixel quanh biên instance.
- Hai object cùng lớp vẫn cần hai instance riêng.
- `instance_id` có dạng `<sample_id>-NNN`; đây là ID trong output lab, không phải class ID hoặc tracking ID.

## Câu hỏi báo cáo (khớp `REPORT.md`)

Trong `REPORT.md`, mỗi mục có dòng **Mở file** + câu hỏi đánh số. Bảng thuật ngữ nằm ở đầu báo cáo.
Trả lời bằng evidence từ JSON/PNG; không chép đáp án có sẵn vì mẫu không còn “gợi ý lộ ý”.

- Mục 1: `traffic` (classification)
- Mục 2–3: `kitchen` (detection + segmentation)
- Mục 4: bảng vòng đời annotator/reviewer
- Mục 5: an toàn dữ liệu
- Mục 6: checklist trước khi nộp link repo

## Xử lý lỗi thường gặp

- **Cài package/tải weight lâu:** chờ ô hiện dấu hoàn tất; không bấm chạy lặp nhiều lần.
- **Colab không cấp GPU:** tiếp tục bằng CPU. Nếu có nguy cơ trễ timeline, báo mentor; không tự đổi model.
- **Tải ảnh lỗi hoặc checksum sai:** chạy lại đúng ô tải ảnh một lần. Nếu vẫn lỗi, chụp thông báo và báo
  mentor; không thay bằng ảnh cá nhân.
- **Không thấy output:** kiểm tra đã chạy đủ các ô trước đó và mở panel Files của Colab.
- **Ô Upload / ZIP báo thiếu hoặc còn trống `REPORT.md`:** điền từ `reports/REPORT_TEMPLATE.md` trên
  máy, chạy lại ô Upload và chọn đúng file đã làm.
- **Chạy lại ô setup giữa chừng:** setup xóa `day1_lab_outputs/`; phải Run all lại từ các ô task.
- **`git status` không thấy bài làm:** kiểm tra hai mục đã nằm trực tiếp dưới `report/`, sau đó chạy
  `git add report`.
- **GitHub thiếu thư mục `visuals/`:** upload cả cây `day1_lab_outputs/`, không chỉ vài file JSON.
- **Số prediction khác bạn bên cạnh:** đối chiếu package, checkpoint, threshold, sample checksum và device
  trước khi kết luận.

## Tự kiểm tra

- [ ] Có evidence từ cả ba JSON và đúng `sample_id`.
- [ ] Có đúng ba PNG trong `day1_lab_outputs/visuals/`.
- [ ] Phân biệt `prediction`, `ground truth`, `class`, `model score`, `box`, `polygon` và `instance`.
- [ ] Không coi confidence/model score là điểm chất lượng nhãn.
- [ ] Nêu ít nhất một lỗi hoặc điểm mơ hồ thực sự quan sát được.
- [ ] Ghi lại mọi thay đổi về code, checkpoint, threshold và môi trường.
- [ ] Có `IMAGE_ATTRIBUTION.md` trong output; họ tên/MSSV chỉ nằm ở tên repository, không nằm trong output.
- [ ] `report/` trên GitHub có trực tiếp `REPORT.md` và `day1_lab_outputs/`.
- [ ] Đã commit, push và nộp **link repository** trên LMS.
