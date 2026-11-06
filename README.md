# Bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

<a href="https://colab.research.google.com/github/VinUni-AI20k/K5-L2-Day1-Data-Overview-AI-ML-DL-Student/blob/v1.0.3/notebooks/day1_understand_labels.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[Mở notebook `day1_understand_labels.ipynb` trên Colab (K5-L2 / v1.0.3)](https://colab.research.google.com/github/VinUni-AI20k/K5-L2-Day1-Data-Overview-AI-ML-DL-Student/blob/v1.0.3/notebooks/day1_understand_labels.ipynb)

> **Trạng thái:** bản phát hành cho lớp; đánh giá formative. Tên repository dùng họ tên và MSSV để Lab
> Coach đối chiếu bài nộp; không ghi thêm dữ liệu cá nhân vào báo cáo hoặc output.
> **Phiên bản lớp:** `v1.0.3`; chỉ đổi release giữa khóa khi GV/Lab Lead thông báo chính thức.
> **Bài nộp:** repository GitHub có thư mục `report/`; chỉ nộp **link repo** trên LMS.
> Nếu Colab mở bản cũ (còn mục Drive ZIP), đóng tab đó và mở lại bằng link `v1.0.3` ở trên, rồi nhấn **Copy to Drive**.

Đây là bài thực hành 60 phút về cách đọc **prediction** của một mô hình đã huấn luyện và suy ra loại
**ground truth** mà một quy trình gán nhãn cần tạo. Bạn không huấn luyện mô hình và không coi prediction
là sự thật.

```text
ảnh công khai → model checkpoint → prediction → đối chiếu guideline → hiểu định dạng ground truth
```

## Phạm vi Ngày 1

- Chạy trên Google Colab; không yêu cầu cài môi trường Python cục bộ.
- Quan sát ba tác vụ: phân loại ảnh, phát hiện vật thể và phân đoạn theo từng đối tượng.
- Đọc JSON, đối chiếu hình minh họa và viết báo cáo có bằng chứng.
- Chỉ dùng ba ảnh COCO công khai đã được notebook cố định và kiểm tra checksum.
- **Không dùng CVAT trong Ngày 1.** Không cài CVAT, không tạo task/job và không nộp annotation CVAT.

## Tương thích Windows, macOS và Ubuntu

Luồng chính chỉ dùng trình duyệt, Google Colab và GitHub Web nên giống nhau trên Windows,
macOS và Ubuntu. Không có lệnh cài đặt hệ thống, đường dẫn ổ đĩa cục bộ hoặc script riêng cho một hệ điều
hành. Chrome và Edge là hai trình duyệt được ưu tiên; Firefox có thể dùng nếu đăng nhập Google/GitHub và
upload thư mục hoạt động bình thường.

## Mục tiêu học tập

Sau bài thực hành, bạn có thể:

1. Phân biệt `prediction` với `ground truth` do con người tạo theo guideline.
2. Đọc tên lớp, mã lớp, model score, tọa độ hộp và đa giác mặt nạ.
3. Phân biệt đơn vị nhãn của phân loại ảnh, detection và instance segmentation.
4. Giải thích class list đến từ taxonomy của dữ liệu/checkpoint, không phải do model tự nghĩ ra.
5. Nêu một lỗi hoặc điểm mơ hồ và đề xuất điều annotator/reviewer cần kiểm tra.

## Ba tác vụ

| Tác vụ | Checkpoint | Taxonomy đi kèm checkpoint | Một prediction mô tả |
| --- | --- | --- | --- |
| Phân loại ảnh | `yolo11n-cls.pt` | ImageNet-1K | một lớp được xếp hạng cho toàn ảnh |
| Phát hiện vật thể | `yolo11n.pt` | COCO, 80 lớp | một lớp và một hộp cho mỗi vật thể được phát hiện |
| Phân đoạn đối tượng | `yolo11n-seg.pt` | COCO, 80 lớp | một lớp, hộp và đa giác cho mỗi instance |

Ba taxonomy/định dạng trên là ví dụ công khai để học cách đọc dữ liệu. Chúng không phải taxonomy hoặc
guideline nội bộ của VinFast.

## Checklist trước khi bắt đầu

Bạn chưa cần cài Python, VS Code, Antigravity hoặc CVAT cho bài này. Trước giờ lab, kiểm tra đủ các mục:

- [ ] Đăng nhập được tài khoản GitHub và tài khoản Google dùng cho Colab.
- [ ] Dùng Chrome hoặc Edge phiên bản mới và có kết nối Internet ổn định.
- [ ] Biết mã khóa (`KHOA`, ví dụ `K4`), họ tên không dấu, MSSV và tên bài (`Data-Overview-AI-ML-DL`) để đặt tên repository bài làm.
- [ ] Mở được repository do GV/Lab Coach cung cấp và nhìn thấy nút **Use this template**.
- [ ] Không sử dụng ảnh cá nhân hoặc dữ liệu nội bộ; notebook đã có sẵn ảnh công khai cho bài lab.
- [ ] Biết kênh hỏi hỗ trợ và kênh nộp **link repository** của lớp.
- [ ] Biết tên repository bắt buộc theo mẫu `<KHOA>-L2-DAY01-HoVaTen-MSSV-Data-Overview-AI-ML-DL`; không thêm email, số điện thoại
  hoặc dữ liệu cá nhân khác.

Nếu thiếu tài khoản hoặc không mở được repository, báo Lab Coach trước khi chạy bài.

## Hướng dẫn từng bước

### Bước 1 — Tạo repository bài làm từ template

Mở repository bằng link do GV/Lab Coach cung cấp, nhấn **Use this template → Create a new repository**.
Chọn tài khoản GitHub của bạn làm owner, đặt tên theo mẫu
`<KHOA>-L2-DAY01-HoVaTen-MSSV-Data-Overview-AI-ML-DL` và chọn **Public** để Lab Coach mở được link nộp.
Thay `<KHOA>` bằng mã khóa của bạn, ví dụ `K4-L2-DAY01-NguyenVanAn-12345678-Data-Overview-AI-ML-DL`.
Chỉ lấy branch mặc định `main`. Nếu đã tạo repository cho bài này thì dùng lại bản đó, không tạo thêm bản sao.

![Repository chính thức với nhãn Public template](docs/screenshots/01-use-template-repository.png)

> Xác nhận đúng tên `VinUni-AI20k/K5-L2-Day1-Data-Overview-AI-ML-DL-Student` và nhãn **Public template** như
> ảnh. Sau khi đăng nhập GitHub, nút **Use this template** nằm ở góc trên bên phải, cạnh Fork/Star.
> Repository tạo từ template có lịch sử độc lập và thư mục `report/` sẵn sàng nhận bài. Không chọn
> **Include all branches**.

### Bước 2 — Mở notebook bằng Google Colab

Quay lại README và nhấn badge **Open in Colab** ở đầu trang. Khi notebook mở ra, nhấn **Copy to Drive**
để tạo bản bạn có thể chỉnh sửa. Nếu Google yêu cầu, đăng nhập và cho phép Colab tạo bản sao trong Drive.

![Notebook trong Colab với nút Run all và Copy to Drive](docs/screenshots/02-open-colab-notebook.png)

> Bạn có thể chạy bằng CPU. Nếu Colab cấp GPU, chọn `Runtime → Change runtime type → T4 GPU`; không có
> GPU không phải là lỗi của bài lab.

### Bước 3 — Chạy notebook từ trên xuống dưới

Nhấn **Run all** hoặc chạy lần lượt từng ô bằng nút `▶`. Không bỏ qua ô cài đặt, tải dữ liệu, kiểm tra
checksum hoặc kiểm tra bằng chứng ở cuối. Nếu runtime bị ngắt giữa chừng, kết nối lại rồi chạy lại từ đầu.

### Bước 4 — Làm `REPORT.md` trên máy, rồi upload trong Colab

Notebook **không** tự sinh báo cáo trống. Trong repo, mở
[`reports/REPORT_TEMPLATE.md`](reports/REPORT_TEMPLATE.md), dựa vào JSON/PNG trong Colab để điền,
lưu thành `REPORT.md` trên máy. Tự đối chiếu [rubric formative](RUBRIC.md).

Trong notebook, chạy ô **Upload `REPORT.md`** và chọn đúng file đã làm xong.

### Bước 5 — Đóng ZIP và đưa lên repo

Chạy ô đóng ZIP (sau khi upload báo cáo). Ô kiểm tra rồi tải về **một** file
`day1_report_for_upload.zip`.

Giải nén → thấy `REPORT.md` và `day1_lab_outputs/` → GitHub repo → thư mục `report/` →
**Add file → Upload files** → kéo hai mục đó vào → **Commit changes**.

ZIP chỉ để chuyển file; **không** nộp ZIP trên LMS và **không** commit ZIP vào repo.

Trên Windows / macOS / Ubuntu chỉ cần trình duyệt; không cần mở terminal.

### Bước 6 — Kiểm tra repo và nộp link

Tải lại trang GitHub, mở `report/` để kiểm tra đủ file rồi nộp **link repository bài làm** trên
LMS. Lab Coach đối chiếu tài khoản LMS với `HoVaTen-MSSV` trong tên repository và chấm nội dung
trong `report/`. Không thêm họ tên, MSSV vào `REPORT.md` hoặc output.

Nếu đã quen Git trên máy: giải nén vào `report/` rồi dùng lệnh ở phần
[Cấu trúc bài nộp](#cấu-trúc-bài-nộp).

Colab miễn phí không bảo đảm GPU hoặc giới hạn tài nguyên cố định. Bản lab dùng biến thể `n` để giảm thời
gian tải/chạy trong lớp; cả ba tác vụ và cấu trúc prediction cần học vẫn giữ nguyên. Không tự đổi checkpoint,
threshold hay mã nguồn; nếu có thay đổi theo hướng dẫn của GV, ghi lại trong báo cáo.

## Cấu trúc bài nộp

Tên repository dùng mẫu `<KHOA>-L2-DAY01-HoVaTen-MSSV-Data-Overview-AI-ML-DL`; thư mục nhận bài luôn là
`report/`. Họ tên và MSSV chỉ xuất hiện trong tên repository để đối chiếu với tài khoản LMS.
Không ghi chúng vào `REPORT.md`, JSON, PNG hoặc attribution và không thêm email, số điện thoại hay dữ
liệu cá nhân khác.

```text
<KHOA>-L2-DAY01-HoVaTen-MSSV-Data-Overview-AI-ML-DL/
└── report/
    ├── REPORT.md
    └── day1_lab_outputs/
        ├── IMAGE_ATTRIBUTION.md
        ├── classification_predictions.json
        ├── detection_predictions.json
        ├── segmentation_predictions.json
        └── visuals/
            ├── classification_top5.png
            ├── detection_predictions.png
            └── segmentation_prediction.png
```

Ví dụ repository: `K4-L2-DAY01-NguyenVanAn-12345678-Data-Overview-AI-ML-DL`. Lab Coach chấm bằng cách mở link repo này.
Nếu LMS quy định khóa/prefix hoặc kênh nộp khác, ưu tiên yêu cầu chính thức.

Lệnh Git tối thiểu sau khi đã chép đúng nội dung:

```bash
git add report
git commit -m "submit day 1 report"
git push
```

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| Notebook chỉ xem được, không sửa hoặc lưu được | Đăng nhập Google rồi nhấn **Copy to Drive**. Làm bài trên bản sao trong Drive của bạn. |
| Không có GPU hoặc Colab báo hết GPU | Tiếp tục bằng CPU. Không tự đổi checkpoint; bài dùng model `n` để vẫn chạy được trên CPU. |
| Ô sau lỗi vì thiếu package/biến/file | Chọn `Runtime → Restart session`, sau đó **Run all** từ đầu; không chỉ chạy lại riêng ô đang lỗi. |
| Tải ảnh/model lỗi hoặc báo sai checksum | Kiểm tra Internet, restart session và chạy lại. Không xóa hoặc bỏ qua bước checksum. |
| Runtime bị ngắt hoặc mất kết nối | Nhấn **Reconnect**, sau đó chạy lại notebook từ đầu để tránh output cũ/thiếu. |
| Không thấy `REPORT.md` hoặc `day1_lab_outputs/` | Chạy lại từ đầu và đọc lỗi ở ô kiểm tra bằng chứng. Không tự tạo file rỗng để thay thế. |
| Ô upload báo thiếu / chưa chọn file | Chạy lại ô Upload và chọn `REPORT.md` đã làm xong trên máy. |
| Ô báo `REPORT.md` còn trùng mẫu trống | Bạn upload nhầm bản chưa điền. Điền xong rồi upload lại. |
| GitHub thiếu JSON/PNG trong `report/` | Upload lại cả thư mục `day1_lab_outputs/`; không chỉ upload `REPORT.md`. |
| GitHub có thêm file ZIP hoặc thư mục bọc | Xóa ZIP/thư mục bọc khỏi `report/`; chỉ giữ `REPORT.md` và `day1_lab_outputs/` trực tiếp. |
| Không upload được thư mục qua GitHub Web | Mở đúng `report/`, kéo thả cả thư mục vào vùng upload. Nếu trình duyệt vẫn chặn, dùng Git hoặc nhờ Lab Coach hỗ trợ. |
| `git push` bị từ chối quyền | Kiểm tra remote đang trỏ tới repository tạo từ template thuộc tài khoản của bạn; đăng nhập lại GitHub nếu cần. |
| Lỡ đưa dữ liệu cá nhân vào báo cáo/output | Giữ định danh bắt buộc ở tên repository, xóa dữ liệu cá nhân khỏi `REPORT.md` và output trước khi nộp link; nếu đã push, báo GV/Lab Coach. |

Nếu lỗi không có trong bảng, chụp màn hình **toàn bộ thông báo lỗi**, ghi rõ bước đang làm và gửi cho Lab
Coach. Không gửi password, token GitHub hoặc mã xác thực Google.

## Quy tắc quan trọng

- `score`/`confidence` là model score dùng để xếp hạng hoặc lọc prediction; không phải điểm chất lượng của
  ground truth và không tự chứng minh prediction đúng.
- `bbox_xyxy` dùng tọa độ pixel `[x_min, y_min, x_max, y_max]`, gốc `(0, 0)` ở góc trên trái.
- `polygon_xy` là các điểm pixel mô tả biên một instance; `instance_id` chỉ nhận diện instance trong bộ
  output của bài lab, không phải class ID hay tracking ID.
- Không sửa nhãn để làm model “đúng”. Hãy ghi nhận sai class, box/mask lệch hoặc vật thể bị bỏ sót.
- Không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab/GitHub công khai.
- Ghi mọi thay đổi về checkpoint, package, threshold và môi trường chạy trong `REPORT.md`.

## Tệp trong repository

| Tệp | Dùng để làm gì |
| --- | --- |
| `notebooks/day1_understand_labels.ipynb` | chạy ba tác vụ, tạo evidence và hướng dẫn đưa bài vào repo |
| `GUIDE.md` | đọc output và xử lý lỗi thường gặp |
| `reports/REPORT_TEMPLATE.md` | mẫu báo cáo; học viên điền rồi upload `REPORT.md` lên Colab |
| `report/` | thư mục học viên đưa `REPORT.md` + `day1_lab_outputs/` vào rồi commit/push |
| `RUBRIC.md` | tự kiểm tra formative; không phải ngưỡng đạt chính thức |
| `requirements.txt` | phiên bản Ultralytics đã ghim cho lần phát hành này |
| `tests/test_repository_contract.py` | kiểm tra tên tệp và hợp đồng notebook |

## Nguồn kỹ thuật

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics Predict mode](https://docs.ultralytics.com/modes/predict/)
- [COCO Terms of Use](https://cocodataset.org/#termsofuse)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

Ba ảnh mẫu dùng trong bản này đều có metadata CC BY 2.0 và notebook tạo `IMAGE_ATTRIBUTION.md` đi kèm
output. Mã nguồn và tài liệu do repository sở hữu được phát hành theo
[AGPL-3.0-only](LICENSE); thành phần bên thứ ba giữ nguyên license riêng trong
[third-party notice](THIRD_PARTY_NOTICES.md). Repository này không tuyên bố phù hợp tiêu chuẩn ISO hoặc
quy trình production của VinFast.
