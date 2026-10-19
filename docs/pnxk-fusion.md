# Kiểm chứng luồng fusion 6 camera + LiDAR

## Kết quả ngày 19/10/2026

Đã hoàn thành một epoch gồm **171 iteration train và 171 frame validation** trên
RTX 5060 Laptop 8 GB, dùng image `bevfusion:rtx5060` và source PNXK hiện tại.
Mục tiêu là kiểm chứng luồng và logic dữ liệu; chưa đánh giá chất lượng detection.

| Thiết lập | Giá trị |
| --- | --- |
| Camera | CAM_P_F, CAM_P_FL, CAM_P_FR, CAM_P_B, CAM_P_LB, CAM_P_RB |
| Tensor ảnh batch | `[1, 6, 3, 256, 704]` |
| LiDAR | LIDAR_TOP, một sweep, point đã đưa về ego |
| Backbone camera | Swin-T, activation checkpointing bật |
| Camera BEV | `[1, 80, 128, 128]` |
| LiDAR BEV | `[1, 256, 128, 128]` |
| ConvFuser output | `[1, 256, 128, 128]` |
| Precision / batch | FP32 / 1 |
| Khởi tạo | Mới, không pretrained |
| Peak memory theo log train | 3.184 MiB |
| Train sau khởi động | Khoảng 1,13 giây/iteration |
| Validation | Khoảng 93 giây |

Loss trung bình nhóm log đầu khoảng 51,3948; nhóm tại iteration 170 khoảng
4,6097. Đây không phải loss trung bình toàn epoch. Train/val đang trùng recording
theo chủ đích debug. Metric vẫn là center-distance diagnostic, không phải IoU AP/NDS.

## Cách chạy lại

Cache fusion `data/pnxk` đã có đủ ảnh undistort và metadata. Từ workspace:

```bash
bash tools/pnxk_docker.sh prepare
bash tools/pnxk_docker.sh check-fusion
bash tools/pnxk_docker.sh audit-fusion
bash tools/pnxk_docker.sh train-fusion
```

`train-fusion` chạy mặc định 1 epoch với `configs/pnxk/fusion.yaml`, launcher none
và run-dir riêng. Có thể chọn run mới hoặc số epoch qua tham số bổ sung:

```bash
bash tools/pnxk_docker.sh train-fusion \
  --run-dir runs/pnxk-fusion-next --max_epochs 2
```

Không chạy audit GPU đồng thời với train trên GPU 8 GB. Các lệnh check/train
LiDAR-only cũ vẫn dùng `check` / `train`.

## Những logic đã kiểm tra

### Dữ liệu và hình học

`tools/check_pnxk.py` kiểm tra trên batch thật của cả train và val:

- Đúng thứ tự sáu camera, đủ sáu ảnh; shape từng ma trận là `[6, 4, 4]`.
- Tâm hình học trong label khớp gravity_center của box framework.
- Vì point cache ở ego: `lidar2ego=I`, `camera2lidar=camera2ego`.
- `camera2lidar × lidar2camera = I` và `lidar2image = K × lidar2camera`.
- K và ảnh cache dùng cùng phép undistort; ImageAug3D lưu phép resize/crop vào
  `img_aug_matrix`, không coi ảnh crop có K gốc mà thiếu post-transform.
- Độ phân giải BEV camera sau downsample khớp grid LiDAR/out_size_factor: 128×128.

Chiều camera extrinsics áp dụng theo kết quả kiểm chứng mẫu: JSON camera là
ego → camera; JSON TOP là sensor → ego. Không transform point cache sang ego lần nữa.
Chi tiết calibration tại [pnxk-data-verification.md](pnxk-data-verification.md).

### Forward/backward GPU thật

`tools/check_pnxk_fusion.py` chạy một batch với model và CUDA pooling thật,
không mock network. Kết quả độc lập lưu trong
[`outputs/pnxk-fusion-audit.json`](../outputs/pnxk-fusion-audit.json):

- Camera encoder: 218 tensor có gradient, gradient hữu hạn và tổng trị tuyệt đối >0.
- LiDAR encoder: 63 tensor có gradient, hữu hạn và tổng trị tuyệt đối >0.
- Fuser: 3 tensor có gradient, hữu hạn và tổng trị tuyệt đối >0.
- Loss hữu hạn; point đầu vào giữ nguyên sau forward/backward.
- Hook xác nhận shape BEV như bảng và thứ tự concat camera trước, LiDAR sau.

Kiểm tra này xác nhận hai nhánh thực sự tham gia tối ưu, không chỉ nạp đủ tên sensor.

### Sửa projection depth

Trong `BaseDepthTransform`:

1. Clone XYZ trước khi undo augmentation để không sửa point dùng bởi LiDAR encoder.
2. Clone camera depth trước clamp, giữ đúng dấu/depth gốc.
3. Loại point sau camera và giá trị không hữu hạn trước khi rasterize.
4. Với scalar depth, chọn point gần nhất ở mỗi pixel thay vì ghi đè phụ thuộc
   thứ tự point trên CUDA. Nếu có point features thì giữ cùng point được chọn.

Test CPU dùng translation augmentation khác zero, hai point chung pixel và một
point sau camera. Bộ 7 tests CPU đã qua sau các thay đổi. CUDA forward/backward
và validation đầy đủ cũng đã chạy qua.

### Bộ nhớ

Lần đầu không activation checkpointing bị thiếu VRAM ở sparse encoder sau nhánh
camera. Đã bật `with_cp: true` trong Swin để recompute activation khi backward.
Giữ nguyên cả sáu camera, ảnh 256×704 và FP32; không dùng camera rỗng để vượt lỗi.

## Artifact và giới hạn

- [Checkpoint epoch 1](../runs/pnxk-fusion-single-gpu/epoch_1.pth), khoảng 460 MB,
  có model và optimizer state.
- [Config đã resolve](../runs/pnxk-fusion-single-gpu/configs.yaml).
- [Train/validation log](../runs/pnxk-fusion-single-gpu/20261019_070242.log).
- [JSON log](../runs/pnxk-fusion-single-gpu/20261019_070242.log.json).
- [Báo cáo audit GPU](../outputs/pnxk-fusion-audit.json).
- [Kiểm tra checkpoint](../outputs/pnxk-fusion-checkpoint.json).

Trong lần đầu, wrapper shell đang chạy bị cập nhật giữa phiên và báo EOF sau
khi Python đã hoàn tất train/validation. Wrapper hiện dùng `exec` cho lệnh cuối
để trả trực tiếp exit code container; cú pháp được kiểm tra bằng `bash -n`.
Audit GPU độc lập và kiểm tra pipeline sau đó đều thoát mã 0.

Các file `runs/` và `outputs/` là artifact cục bộ, được gitignore. Kiểm chứng
hiện giới hạn trong recording/chunk và calibration mẫu, chưa xác nhận toàn bộ xe/ngày,
multi-GPU hay chất lượng model. Với recording mới vẫn cần kiểm tra calibration,
timestamp và ảnh projection trước khi train chính thức.
