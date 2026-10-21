# Adapt BEVFusion cho PNKX

> **Cập nhật profile nuScenes:** Các thông số/kết quả bên dưới mô tả bản debug
> BEV 128 cũ. Xem [profile mới và cách chạy](pnkx-nuscenes-profile.md) cho BEV 180,
> FP16, CBGS, augmentation và lịch train theo nuScenes. Đã sửa thêm quy ước
> LWH/yaw → WLH/yaw legacy ở dataset; checkpoint debug cũ cần train lại.
> Các lệnh wrapper có hậu tố `-debug` dưới đây dùng config debug hiện hành.


Tên chuẩn hiện tại là **PNKX**. Các kết quả train/verify ngày 19/10/2026 bên dưới
là lịch sử của mẫu VF6_01. Converter hiện đã map camera sau VF6_03 từ
`CAM_P_B_New`, dùng calibration cùng tên; xem profile mới bên dưới.
Xem [đối chiếu PNKX/nuScenes và trạng thái split mới](pnkx-vs-nuscenes.md).
Các link `runs/pnxk-*`, `outputs/pnxk-*` được giữ để trỏ đúng artifact lịch sử.

Giải thích từng bước, input/output và lý do sửa code:
[pnkx-pipeline-explained.md](pnkx-pipeline-explained.md).

## Trạng thái triển khai

Đã thêm converter, PNKXDataset, config LiDAR-only/fusion, kiểm tra cache và
evaluator chẩn đoán. Bộ mẫu đã chuyển đổi: 171 sample train, 171 sample val,
6 camera/sample, không bỏ frame hay chunk. Cache hiện tại khoảng 1,6 GB.

Đã tìm thấy image Docker `bevfusion:rtx5060` có stack phù hợp RTX 5060.
Kiểm tra ngoài sandbox xác nhận driver hoạt động; lỗi NVIDIA-SMI trước đó do
quyền truy cập thiết bị trong sandbox. Đã nạp/collate batch train và val thật
qua MMDetection. Đã hoàn thành 1 epoch train (171 iteration) và validation 171 frame,
lưu checkpoint tại `runs/pnkx-lidar-single-gpu/epoch_1.pth`.
Chi tiết môi trường và cách chạy tại [pnkx-environment.md](pnkx-environment.md).

## 1. Quy tắc dữ liệu

- TXT quyết định split ở mức recording. Converter duyệt mọi chunk trong recording.
- Chấp nhận recording name hoặc đường dẫn cũ kết thúc bằng `data/<recording>`.
- Mặc định chặn recording trùng train/val. `--allow-split-overlap` chỉ dành cho debug.
- Label timestamp làm mốc. TOP phải lệch tối đa 1 ms, camera tối đa 40 ms mặc định.
  Hai ngưỡng có thể điều chỉnh bằng `--lidar-max-delta-ms`, `--camera-max-delta-ms`.
- Thiếu label, sensor, calibration hoặc vượt ngưỡng timestamp sẽ báo lỗi.
  Có thể chủ động bỏ chunk không label bằng `--skip-missing-labels`, bỏ frame
  không ghép được timestamp bằng `--skip-unmatched`; báo cáo lưu lý do.
- VF6_01 chọn calibration theo ngày recording với khoảng `[valid_from, valid_to)`;
  không tự lấy calibration gần nhất khi ngày không thuộc khoảng nào.
- Giai đoạn này dùng một sweep LIDAR_TOP. Không ghép LiDAR phụ, radar hay sweep lịch sử.
- Cache point gồm float32 `[x_ego, y_ego, z_ego, intensity_raw, sweep_dt=0]`.
  Giữ intensity gốc; không giảm beam. Point TOP đã transform sensor → ego.
- Box cache giữ tâm hình học và LWH/yaw raw. Dataset đổi sang WLH và
  `yaw_internal=-yaw_raw-π/2` theo quy ước legacy của repo, rồi dùng
  `origin=(0.5, 0.5, 0.5)` để chuyển sang tâm đáy. Không cộng velocity giả.
- Class order cố định: `car`, `rider`, `pedestrian`.
- Giữ box có annotation point-count bằng 0. Trường count được lưu tham khảo,
  chưa dùng để lọc hay coi là số điểm TOP đã tính lại.

Chiều extrinsics và giới hạn kiểm chứng được ghi trong
[pnkx-data-verification.md](pnkx-data-verification.md).

## 2. Convert

Dependency riêng cho tiền xử lý: NumPy, laspy, lazrs; thêm OpenCV cho camera.
Trong phiên triển khai đã cài thư viện tạm vào `/tmp/pnkx-inspect-deps`, chạy:

```bash
PYTHONPATH=/tmp/pnkx-inspect-deps python tools/data_converter/pnkx_converter.py \
  --root Data-PNKX \
  --train-split Data-PNKX/split/train.txt \
  --val-split Data-PNKX/split/val.txt \
  --output data/pnkx \
  --allow-split-overlap \
  --with-cameras \
  --camera-extrinsics ego-to-camera \
  --distortion-model opencv-standard
```

Camera theo thứ tự F, FL, FR, B, LB, RB của nhóm `CAM_P_*`. Ảnh cache được
undistort bằng mô hình chuẩn OpenCV và giữ K gốc. Hai tùy chọn camera bắt buộc
ghi rõ giả định; `ego-to-camera` đã được kiểm chứng trên chunk mẫu, chưa xác nhận
cho mọi xe/ngày. Không có fallback sang camera khác khi calibration thiếu.

Muốn convert LiDAR-only thì bỏ ba tùy chọn `--with-cameras`,
`--camera-extrinsics`, `--distortion-model`. Nên dùng output khác, ví dụ
`data/pnkx-lidar`, để phân biệt hai cache. Cache fusion dùng được cho cả hai config.

Để kiểm tra nhanh, thêm `--max-frames-per-chunk 3` và chọn output riêng.
Khi train thật bỏ `--allow-split-overlap` và giới hạn số frame.

Converter ghi đè các file cache tương ứng khi chạy lại. Dữ liệu gốc không bị sửa.
Ảnh/point cũ không còn được metadata tham chiếu có thể vẫn nằm trong output;
metadata mới quyết định sample nào được nạp.

Đầu ra:

```text
data/pnkx/
├── pnkx_infos_train.pkl
├── pnkx_infos_val.pkl
├── conversion_report.json
├── points/<recording>/<chunk>/<timestamp>.bin
└── images/<recording>/<chunk>/<camera>/<timestamp>.jpg
```

Đường dẫn point/ảnh trong metadata tương đối với cache root, có thể chuyển cả
thư mục cache sang máy khác rồi cập nhật `dataset_root` trong config.
Metadata PKL phải do nguồn tin cậy tạo vì pickle không an toàn với file lạ.
Metadata chỉ lưu kiểu Python cơ bản; dataset tạo lại mảng NumPy khi nạp, tránh
phụ thuộc đường dẫn module pickle khác nhau giữa NumPy 2 của máy convert và
NumPy cũ của môi trường train.

## 3. Kiểm tra trước train

Kiểm tra cache trên CPU, không cần mmcv/CUDA:

```bash
python tools/check_pnkx.py --root data/pnkx
PYTHONPATH=/tmp/pnkx-inspect-deps python -m unittest discover -s tests -v
```

Checker đọc toàn bộ point/ảnh của các split, kiểm tra schema, ID, class,
số chiều point, single-sweep time lag và ma trận camera khả nghịch.

Trong môi trường train đúng dependency, kiểm tra dataset/pipeline/collate thật:

```bash
python tools/check_pnkx.py --root data/pnkx --pipeline-config configs/pnkx/lidar.yaml
python tools/check_pnkx.py --root data/pnkx --pipeline-config configs/pnkx/fusion.yaml
```

Pipeline LiDAR-only đã chạy thành công trong container cho train và val.
Python base trên host vẫn thiếu dependency; dùng wrapper Docker ở tài liệu
môi trường để chạy. Fusion cũng đã hoàn thành train/validation; xem
[pnkx-fusion.md](pnkx-fusion.md) để chạy và xem các kiểm tra logic.

Do đã thay đổi cấu trúc cache, có thêm kiểm tra tích hợp convert nhiều recording,
mỗi recording nhiều chunk, để bảo đảm split không bị chia lại theo frame/chunk
và point được transform sang ego đúng một lần.

Bảy kiểm tra CPU đã chạy qua gồm timestamp/ngưỡng ghép, split/leakage,
parse box/class, khoảng calibration không fallback, matching AP một-một,
convert nhiều recording/chunk và interface DepthLSS. Test DepthLSS chạy projection/geometry/convolution Torch
thật trên CPU, thay registry/decorator và CUDA BEV pooling bằng phần giả lập;
không chứng minh CUDA pooling hay toàn bộ model hoạt động.

## 4. Config và chạy train

- `configs/pnkx/default.yaml`: model chung, dataset và pipeline debug.
- `configs/pnkx/lidar.yaml`: chỉ LiDAR.
- `configs/pnkx/fusion.yaml`: sáu camera + LiDAR, Swin-T + DepthLSS + ConvFuser.

Config dùng cơ chế load recursive của torchpack giống repo gốc. Mặc định batch 1,
workers 0, 2 epoch, FP32, tắt augmentation để kiểm tra luồng ban đầu. Phạm vi
ego `[-51.2, -51.2, -5, 51.2, 51.2, 3]`, voxel `[0.1, 0.1, 0.2]`.
Train lọc box theo ObjectRangeFilter; evaluator lọc tâm box theo phạm vi 3D.

Head TransFusion có 3 class và code 8 chiều (center, height, log-dim, sin/cos yaw),
không có velocity. Xử lý peak cho rider/pedestrian dùng chỉ số class PNKX;
không truy cập class 8/9 của nuScenes. NMS đang tắt theo cấu hình baseline.

Không sử dụng database sampling, map segmentation, radar, CBGS hoặc multi-sweep.
PNKXCollect3D cấp tensor rỗng cho đối số ảnh/depth không dùng trong chế độ LiDAR,
để khớp chữ ký forward chung của repo. Trong fusion, point cache ở ego nên
`lidar2ego=I`, `camera2lidar=camera2ego`, `lidar2camera=ego2camera`.

Lệnh chạy sau khi môi trường GPU sẵn sàng:

```bash
torchpack dist-run -np 1 python tools/train.py configs/pnkx/lidar.yaml \
  --run-dir runs/pnkx-lidar-debug

torchpack dist-run -np 1 python tools/train.py configs/pnkx/fusion.yaml \
  --run-dir runs/pnkx-fusion-debug
```

Trên máy RTX 5060 hiện tại, dùng chế độ một GPU trực tiếp để tránh bước
khởi tạo MPI/NCCL đang bị kẹt:

```bash
bash tools/pnkx_docker.sh prepare
bash tools/pnkx_docker.sh check-debug
bash tools/pnkx_docker.sh train-debug
```

Wrapper chạy `tools/train.py --launcher none`. Launcher MPI mặc định của repo
vẫn được giữ cho cấu hình distributed; chưa kiểm chứng multi-GPU trên máy này.

Các config hiện khởi tạo mới, không tự tải checkpoint. Đây là baseline kiểm tra
pipeline; chưa chốt lịch fine-tune/pretrained cho train chính thức.
Khi chọn checkpoint cần kiểm tra input features, voxel/grid, số class và tensor
head không tương thích trước khi nạp. Không dùng resume cho checkpoint khác cấu hình.

## 5. Metric hiện có

Dataset hỗ trợ `center_distance` để xác nhận vòng train/validation trên data riêng,
không gọi nuScenes SDK evaluator:

- Match cùng class, cùng frame, greedy theo score giảm dần, một GT chỉ match một lần.
- Dùng khoảng cách tâm XY tại ngưỡng 0,5 / 1 / 2 / 4 m.
- AP là diện tích precision-recall nội suy all-points.
- mAP ở mỗi ngưỡng chỉ trung bình các class có GT; trả thêm số GT từng class.
- Key ví dụ: `center_AP/car@2.0m`, `center_mAP@2.0m`.

Metric này bỏ qua yaw và kích thước box, không phải IoU 3D AP hoặc nuScenes NDS.
Train/val trùng recording hiện tại chỉ kiểm tra pipeline/overfit. Cần bổ sung
đánh giá IoU 3D và split độc lập trước khi đánh giá chất lượng thực tế.

## 6. Sửa tương thích nhánh camera

`DepthLSSTransform` của repo gọi base depth transform đang truyền `mats_dict`,
nhưng hàm con trước đây chỉ nhận `(x, d)`. Đồng thời default của base bật thêm
point features và height expansion trong khi convolution của DepthLSS chỉ nhận
một kênh depth. Đã sửa riêng subclass này:

- `height_expand=False`, `add_depth_features=False`.
- `get_cam_feats(x, d, mats_dict=None)` nhận được interface base hiện tại.

Các transform khác giữ default riêng. Thay đổi này có test CPU như mô tả ở mục 3.
