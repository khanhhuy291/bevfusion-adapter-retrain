# Môi trường PNKX LiDAR-only trên RTX 5060

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

## Môi trường đã xác nhận ngày 19/10/2026

| Thành phần | Giá trị |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8.151 MiB |
| Driver | 595.99.02 |
| Image Docker có sẵn | `bevfusion:rtx5060` |
| Image ID | `sha256:fe2e7a84bcf5dded747d45971c8b5eedd6270156de3af3b115b2a134ec61de09` |
| Python trong image | 3.10.12 |
| PyTorch | 2.7.1+cu128 |
| CUDA runtime của Torch | 12.8 |
| MMCV | 1.7.2 |
| MMDetection | 2.28.2 |
| NumPy | 1.23.5, theo constraints trong image |

GPU/driver hoạt động khi kiểm tra ngoài sandbox. Python base của host không
phải môi trường train; không cài đè dependency vào base và không thay driver.

Image này đã tồn tại trên máy trước bước triển khai. Nó có dependency và
13 extension CUDA build sẵn. Quy trình hiện tại tái sử dụng image này, không
khẳng định Dockerfile cũ CUDA 11.3 của repo build được cho máy RTX 5060.
Blackwell cần stack có hỗ trợ kiến trúc GPU tương ứng; tham khảo
[NVIDIA CUDA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
và [thông tin hỗ trợ Blackwell của PyTorch](https://discuss.pytorch.org/t/request-for-sm-120-rtx-50xx-cuda-12-8-support-in-pytorch/222207/2).

## Chuẩn bị và kiểm tra

Chạy từ checkout này, với quyền truy cập Docker và NVIDIA Container Toolkit:

```bash
nvidia-smi
docker images
bash tools/pnkx_docker.sh prepare
bash tools/pnkx_docker.sh check-debug
```

`prepare` chép extension `.so` từ `/workspace/bevfusion-adapter/mmdet3d/ops`
trong image vào thư mục tương ứng của checkout. Các file này được gitignore.
Chỉ dùng chúng với Python/Torch/CUDA ABI của image đã tạo ra chúng; chạy lại
prepare khi thay image. Binaries không phải phần source đã commit.

`check` mount checkout vào `/work`, đặt `PYTHONPATH=/work`, dùng UID/GID của
người chạy để file output không thuộc root. Điều này bảo đảm import code PNKX
hiện tại, không vô tình train code cũ được đóng sẵn trong image.

Kết quả kiểm tra batch đầu: 172.514 point trong phạm vi train, tensor point
5 chiều, tensor ảnh rỗng `(0, 3, 1, 1)` vì LiDAR-only. Train và val collate đều qua.

Có thể chọn image bằng biến `PNKX_IMAGE`, nhưng image thay thế phải có cùng
cấu trúc thư mục extension và dependency phù hợp. Máy khác cần có image này
hoặc build lại môi trường tương đương; wrapper không tự tải/build image.

## Train một GPU

```bash
bash tools/pnkx_docker.sh train-debug
```

Lệnh mặc định chạy 1 epoch, batch 1, FP32, từ khởi tạo mới trên 171 frame,
validation 171 frame. Train/val hiện trùng recording theo chủ đích debug.
Output: `runs/pnkx-lidar-single-gpu/`.

Có thể thêm tham số cuối, ví dụ chọn run mới và 2 epoch:

```bash
bash tools/pnkx_docker.sh train-debug --run-dir runs/pnkx-lidar-debug-2epochs --max_epochs 2
```

Wrapper dùng `--launcher none`, không qua MPI hoặc process group NCCL.
Lần thử launcher MPI bị kẹt trước iteration đầu và đã được dừng; chưa xác
định nguyên nhân gốc của NCCL, không coi đó là lỗi GPU hay dữ liệu.

## Kết quả chạy kiểm chứng

- Hoàn thành 171 iteration train và 171 frame validation; tiến trình thoát mã 0.
- Batch 1, FP32, một sweep TOP, khởi tạo mới; không dùng pretrained.
- Loss trung bình nhóm 10 iteration đầu: 46,4025; nhóm được log ở iteration 170:
  5,0335. Đây không phải trung bình loss của toàn epoch.
- Sau khởi động, train khoảng 0,31 giây/iteration. Validation khoảng 23 giây.
- Bộ nhớ CUDA đỉnh do PyTorch/MMCV log: 1.401 MiB, không phải toàn bộ VRAM hệ thống.
- `center_mAP@2.0m` khoảng 0,00004; `center_mAP@4.0m` khoảng 0,00028
  (thang 0–1 theo JSON log). Chưa có chất lượng phát hiện tốt sau một epoch.
- Train/val trùng recording, metric chỉ phục vụ kiểm tra pipeline; không đánh giá
  khả năng tổng quát hóa. Cần kiểm tra prediction trực quan và khả năng overfit
  trước khi mở rộng train hoặc kết luận về chất lượng.
- Checkpoint: [`epoch_1.pth`](../runs/pnxk-lidar-single-gpu/epoch_1.pth), khoảng 96 MB.
  Đã đọc lại trên CPU: epoch 1, iteration 171, 298 tensor trong state_dict,
  không có trọng số floating-point NaN/Inf. Có lưu optimizer state.
- Log: [`20261019_024624.log`](../runs/pnxk-lidar-single-gpu/20261019_024624.log),
  [`JSON log`](../runs/pnxk-lidar-single-gpu/20261019_024624.log.json).
  Tên/log timestamp dùng UTC của container; tương ứng khoảng 09:46–09:47 giờ Việt Nam.
- Config đã resolve: [`configs.yaml`](../runs/pnxk-lidar-single-gpu/configs.yaml).

Checkpoint/log nằm trong `runs/` đã gitignore, không tự đi kèm source khi clone repo.

## Thay đổi tương thích đã áp dụng

1. `tools/train.py`: thêm `--launcher none`; mặc định `mpi` vẫn giữ nguyên.
2. `mmdet3d/apis/train.py`: nhánh single GPU dùng MMDataParallel, loader một GPU,
   EvalHook; nhánh distributed vẫn dùng MMDistributedDataParallel/DistEvalHook.
3. `mmdet3d/utils/torch_compat.py`: MMCV 1.x truyền CUDA ID dạng int cho helper
   scatter riêng của Torch mới. Shim chuyển sang `torch.device` cho Torch >=2.1,
   cài một lần trong train_model. Không sửa site-packages của image.
4. `mmdet3d/ops/spconv/conv.py`: force đăng ký lớp convolution đi kèm repo,
   tránh trùng registry với MMCV 1.7; giữ implementation/weight layout của repo.
5. `radar_encoder.py`: bỏ import FlashMHA không được dùng, để import model
   không bắt buộc cài FlashAttention khi chỉ chạy LiDAR.

Hai thay đổi cuối đã có bản tương ứng trong image; đã đối chiếu rồi áp dụng
vào source hiện tại. Không chép đè dataset/model từ image lên adapter PNKX.
