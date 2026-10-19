# Hướng dẫn train fusion PNXK từ raw đến checkpoint

Hướng dẫn này dành cho workspace hiện tại trên máy RTX 5060 Laptop 8 GB,
dùng image Docker **`bevfusion:rtx5060` đã có sẵn**. Quy trình đã chạy qua với
171 frame mẫu: convert → kiểm tra → train một epoch → validation → checkpoint.

Mục tiêu là xác nhận luồng train camera + LiDAR chạy thông và đúng logic trên
dữ liệu mẫu. Train/val hiện trùng recording để debug, chưa dùng đánh giá chất lượng.

## 1. Vào workspace và kiểm tra môi trường

Chạy toàn bộ lệnh từ thư mục gốc repo:

```bash
cd /home/khanhhuy/bevfusion-adapter-retrain
nvidia-smi
docker image inspect bevfusion:rtx5060 --format '{{.Id}}'
```

Máy đã kiểm chứng dùng Python 3.10, Torch 2.7.1+cu128, CUDA 12.8,
MMCV 1.7.2 và MMDetection 2.28.2 **bên trong container**. Không dùng Python base
của host để chạy model và không cài đè dependency train lên base.

Docker phải truy cập được GPU. Image trên là điều kiện có sẵn của hướng dẫn;
script không tự tải hoặc build lại image. Thông tin đầy đủ tại
[pnxk-environment.md](pnxk-environment.md).

Trong phiên Codex, kiểm tra GPU/Docker cần quyền ngoài sandbox. Nếu NVIDIA-SMI
báo lỗi trong sandbox, kiểm tra lại trong terminal của máy trước khi kết luận driver hỏng.

## 2. Chuẩn bị raw và label

```text
Data-PNXK/
├── data/
│   └── <recording>/
│       ├── <chunk-1>/
│       │   ├── CAMERA/
│       │   │   ├── CAM_P_F/*.jpg
│       │   │   ├── CAM_P_FL/*.jpg
│       │   │   ├── CAM_P_FR/*.jpg
│       │   │   ├── CAM_P_B/*.jpg
│       │   │   ├── CAM_P_LB/*.jpg
│       │   │   └── CAM_P_RB/*.jpg
│       │   └── LIDAR/LIDAR_TOP/*.laz
│       └── <chunk-2>/...
├── label/
│   └── <recording>/
│       ├── <chunk-1>/*.txt
│       └── <chunk-2>/*.txt
├── split/
│   ├── train.txt
│   └── val.txt
├── VF6_01_Intrinsics.json
└── VF6_01_Extrinsics_By_Dates.json
```

Các sensor khác có thể giữ nguyên trong raw; config hiện dùng **6 CAM_P + TOP**,
chưa dùng radar hoặc bốn LiDAR phụ. File calibration phải tương ứng xe/ngày recording.

Mỗi dòng label:

```text
x y z length width height yaw class occlusion_level number_of_point track_id
```

Quy ước đã chốt:

- Box ở ego: x trước, y trái, z lên; mét; tâm hình học.
- Yaw radian, 0 hướng +x, chiều dương từ +x sang +y.
- Class raw: `Car`, `Rider`, `Pedestrian`; converter đổi về chữ thường.
- Point raw LAZ ở hệ sensor. Extrinsics TOP dùng sensor → ego.
- Với calibration mẫu đã thử, camera JSON dùng **ego → camera**.
- Ảnh raw chưa undistort. Converter sẽ xử lý méo, không cần sửa ảnh gốc.

Chiều extrinsics camera không được suy rộng cho xe/ngày khác nếu chưa kiểm chứng.
Xem [quy trình kiểm chứng hình học](pnxk-data-verification.md).

## 3. Chuẩn bị split theo recording

Mỗi dòng trong `Data-PNXK/split/train.txt` hoặc `val.txt` là một recording:

```text
20260228_1105_VF6_01_1772251511_1772254880
```

Converter cũng chấp nhận dạng đường dẫn cũ:

```text
PNKX-26_09_2026/data/20260228_1105_VF6_01_1772251511_1772254880
```

Converter lấy recording, duyệt tất cả chunk của recording đó và dùng các frame
có label. **Không chia ngẫu nhiên lại theo chunk hoặc frame.**

Với bộ test hiện tại, hai file cùng recording: phải thêm `--allow-split-overlap`
khi convert. Với train thật, tách recording giữa hai split và bỏ tùy chọn này.

## 4. Chuẩn bị thư viện convert trên host

Tiền xử lý không cần GPU. Trong máy hiện tại, Python host đã có NumPy và Pillow;
các thư viện bổ sung từng được cài vào `/tmp/pnxk-inspect-deps`:

```bash
python -c "import numpy; from PIL import Image; print('NumPy/Pillow OK')"
PYTHONPATH=/tmp/pnxk-inspect-deps python -c "import laspy, lazrs, cv2; print('LAZ/OpenCV OK')"
```

Nếu thư mục tạm đã bị dọn, cài lại các phiên bản đã dùng:

```bash
python -m pip install --target /tmp/pnxk-inspect-deps --no-deps \
  laspy==2.7.0 lazrs==0.8.2 opencv-python-headless==5.0.0.93
```

Lệnh `--no-deps` trên dựa vào NumPy/Pillow sẵn có của máy hiện tại. Trên máy khác,
cần chuẩn bị Python tiền xử lý có đủ NumPy, Pillow và các thư viện trên trước.

## 5. Convert raw thành cache fusion

```bash
PYTHONPATH=/tmp/pnxk-inspect-deps python tools/data_converter/pnxk_converter.py \
  --root Data-PNXK \
  --train-split Data-PNXK/split/train.txt \
  --val-split Data-PNXK/split/val.txt \
  --output data/pnxk \
  --allow-split-overlap \
  --with-cameras \
  --camera-extrinsics ego-to-camera \
  --distortion-model opencv-standard
```

Converter thực hiện:

1. Kiểm tra recording thuộc split và calibration theo xe/ngày.
2. Lấy timestamp label làm mốc; ghép TOP và ảnh camera gần nhất.
3. Đọc LAZ có áp dụng scale/offset, transform point TOP sang ego.
4. Ghi point float32 `[x, y, z, intensity_raw, sweep_dt=0]`.
5. Undistort sáu camera, giữ K và lưu ảnh cache riêng.
6. Ghi metadata train/val, box tâm hình học và thông tin calibration đã chọn.

Mặc định TOP lệch tối đa 1 ms, camera tối đa 40 ms. Thiếu sensor/calibration,
thiếu label chunk hoặc vượt ngưỡng sẽ báo lỗi. Không bật các tùy chọn skip khi
chưa hiểu lý do frame bị bỏ; nếu chủ động skip thì xem report.

Kết quả với mẫu hiện tại:

```text
train: 171 samples, 0 skipped frames, 0 skipped chunks
val: 171 samples, 0 skipped frames, 0 skipped chunks
```

```text
data/pnxk/
├── pnxk_infos_train.pkl
├── pnxk_infos_val.pkl
├── conversion_report.json
├── points/<recording>/<chunk>/*.bin
└── images/<recording>/<chunk>/<camera>/*.jpg
```

Cache mẫu khoảng 1,6 GB. Converter không sửa raw nhưng sẽ ghi đè cache cùng tên
khi chạy lại. Khi raw/split/calibration thay đổi, cần convert lại trước khi train.

## 6. Chuẩn bị extension CUDA và kiểm tra batch

```bash
bash tools/pnxk_docker.sh prepare
bash tools/pnxk_docker.sh check-fusion
```

`prepare` lấy 13 extension CUDA build sẵn trong image cho đúng Python/Torch ABI.
`check-fusion` mount source hiện tại vào container, đọc cache và kiểm tra logic
dataset/pipeline thật. Không train code cũ đóng sẵn trong image.

Kết quả mong đợi trên mẫu:

```text
train pipeline/collate OK: 172514 points, img (6, 3, 256, 704)
val pipeline/collate OK: 172514 points, img (6, 3, 256, 704)
```

Checker cũng kiểm tra thứ tự camera, tâm box, `lidar2ego=I`, cặp transform
nghịch đảo, `lidar2image=K×lidar2camera` và kích thước BEV của hai nhánh.
Số điểm sẽ khác nếu dùng dữ liệu khác.

## 7. Kiểm tra một forward/backward fusion thật

```bash
bash tools/pnxk_docker.sh audit-fusion
```

Audit dùng GPU thật, không mock network. Kết quả cần có:

- Tensor ảnh `[1,6,3,256,704]`.
- Camera BEV `[1,80,128,128]`.
- LiDAR BEV và fused BEV `[1,256,128,128]`.
- Loss hữu hạn.
- Camera, LiDAR và fuser đều có gradient hữu hạn, `absolute_sum > 0`.
- `points_unchanged: true`.

Báo cáo lưu tại `outputs/pnxk-fusion-audit.json`. Chạy audit xong rồi mới train,
không chạy đồng thời hai tiến trình GPU trên máy 8 GB.

## 8. Train fusion một epoch

```bash
bash tools/pnxk_docker.sh train-fusion
```

Lệnh này tương đương chạy trong container:

```bash
python tools/train.py configs/pnxk/fusion.yaml \
  --launcher none \
  --run-dir runs/pnxk-fusion-single-gpu \
  --max_epochs 1
```

Lệnh `python` tương đương chỉ chạy **trong môi trường container**, không dùng
Python base của host. Wrapper đã cấu hình GPU, shared memory, UID/GID và PYTHONPATH.

Cấu hình hiện tại:

- Sáu ảnh CAM_P kích thước 256×704, TOP single sweep.
- Swin-T + DepthLSS, SparseEncoder, ConvFuser và head TransFusion 3 class.
- Batch 1, FP32, activation checkpointing Swin bật để vừa VRAM 8 GB.
- Không radar, không map segmentation, không database sampling, không velocity giả.
- Khởi tạo mới, không tự tải pretrained; không phải resume từ lần train LiDAR-only.

Muốn dùng run-dir khác để giữ nguyên kết quả cũ:

```bash
bash tools/pnxk_docker.sh train-fusion \
  --run-dir runs/pnxk-fusion-test-02 --max_epochs 1
```

Muốn chạy nhiều epoch từ khởi tạo mới, đổi `--max_epochs`. Tăng số epoch không
tự resume checkpoint. Dùng run-dir riêng cho từng lần thử.

## 9. Xác nhận train thành công

Một lần chạy thành công cần đủ các dấu hiệu sau:

1. Log đi qua các iteration train, loss hữu hạn và không có traceback.
2. Có dòng `Saving checkpoint at 1 epochs`.
3. Validation đi hết frame và log `Epoch(val) [1][171]` với bộ mẫu hiện tại.
4. Tiến trình kết thúc thành công, có checkpoint và config đã resolve.

```bash
ls -lh runs/pnxk-fusion-single-gpu/epoch_1.pth
ls -l runs/pnxk-fusion-single-gpu/latest.pth
ls runs/pnxk-fusion-single-gpu/*.log
```

Các file đầu ra:

```text
runs/pnxk-fusion-single-gpu/
├── configs.yaml
├── <timestamp>.log
├── <timestamp>.log.json
├── epoch_1.pth
└── latest.pth -> epoch_1.pth
```

Log cuối lần kiểm chứng nằm tại
[`20261019_070242.log`](../runs/pnxk-fusion-single-gpu/20261019_070242.log).
Checkpoint đã có tại [`epoch_1.pth`](../runs/pnxk-fusion-single-gpu/epoch_1.pth):
epoch 1, iteration 171, khoảng 460 MB, có cả camera encoder/LiDAR encoder/fuser
và optimizer state. Đã đọc lại 574 tensor state_dict, không có trọng số NaN/Inf.

Lần kiểm chứng dùng khoảng 3.184 MiB bộ nhớ đỉnh theo log PyTorch;
train khoảng 1,13 giây/iteration sau khởi động và validation khoảng 93 giây.
Các số này chỉ là quan sát trên máy/bộ mẫu hiện tại.

Lần chạy đầu từng báo EOF ở wrapper sau khi train/val đã xong do file script
bị cập nhật khi đang thực thi. Wrapper hiện dùng `exec` để trả exit code container;
cú pháp và lệnh check-fusion sau sửa đã kiểm tra thành công. Tránh sửa script
hoặc source model trong khi một lần train đang chạy.

## 10. Khi gặp lỗi

| Hiện tượng | Kiểm tra/xử lý |
| --- | --- |
| Không thấy image `bevfusion:rtx5060` | Cần chuẩn bị image tương đương trước; hướng dẫn này không build image từ đầu |
| Docker permission denied | Dùng terminal/tài khoản có quyền Docker; trong Codex cần quyền ngoài sandbox |
| Không import được extension `.so` | Chạy `prepare`; dùng cùng image/Python ABI, không dùng host Python để import model |
| Train/val overlap khi convert | Với mẫu debug dùng `--allow-split-overlap`; train thật sửa split theo recording |
| Không có calibration cho ngày/xe | Bổ sung đúng calibration, không lấy tùy ý khoảng gần nhất |
| Nearest sensor vượt ngưỡng timestamp | Xem frame, timestamp và sensor báo lỗi trước khi đổi ngưỡng |
| Thiếu VRAM khi fusion | Xác nhận batch 1 và `with_cp: true` trong config fusion; dừng tiến trình GPU khác |
| MPI/NCCL bị kẹt | Dùng wrapper với `--launcher none` trên máy một GPU hiện tại |
| AP thấp dù train chạy xong | Không đồng nghĩa pipeline lỗi; bước này mới kiểm chứng luồng, chưa tối ưu chất lượng |

## 11. Chuỗi lệnh ngắn cho lần chạy từ raw

Sau khi đã có image và thư viện convert:

```bash
cd /home/khanhhuy/bevfusion-adapter-retrain

PYTHONPATH=/tmp/pnxk-inspect-deps python tools/data_converter/pnxk_converter.py \
  --root Data-PNXK \
  --train-split Data-PNXK/split/train.txt \
  --val-split Data-PNXK/split/val.txt \
  --output data/pnxk \
  --allow-split-overlap \
  --with-cameras \
  --camera-extrinsics ego-to-camera \
  --distortion-model opencv-standard

bash tools/pnxk_docker.sh prepare
bash tools/pnxk_docker.sh check-fusion
bash tools/pnxk_docker.sh audit-fusion
bash tools/pnxk_docker.sh train-fusion --run-dir runs/pnxk-fusion-raw-test
```

Mỗi lệnh cần thành công trước khi chạy lệnh tiếp theo. Với run-dir cuối ở ví dụ,
checkpoint nằm trong `runs/pnxk-fusion-raw-test/`.

Raw, cache, checkpoint và audit report được gitignore, không tự đi kèm source
khi clone repo. Phạm vi đã xác minh là recording/chunk/calibration mẫu; dữ liệu
xe/ngày khác vẫn cần kiểm chứng projection trước khi mở rộng train.
