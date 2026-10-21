# PNKX: cấu hình theo baseline nuScenes

Profile mới nằm trong `configs/pnkx/nuscenes/`. Mục tiêu là khớp cấu trúc model,
lưới không gian và recipe train của baseline trong repo, đồng thời giữ các điều
chỉnh cần thiết cho label và sensor PNKX. Raw và hai file split của người dùng
không bị sửa. Các config `configs/pnkx/{lidar,fusion}.yaml` vẫn là bản debug cũ.

**Mapping camera hiện tại:** theo xác nhận của người dùng, converter dùng
`CAM_P_B_New` làm camera sau cho VF6_03, cùng intrinsic/extrinsic key
`CAM_P_B_New`. VF6_01 vẫn dùng `CAM_P_B`. Sáu slot metadata/model giữ thứ tự
`F, FL, FR, B, LB, RB`; slot `CAM_P_B` có `source_name` và `source_path` ghi sensor
và ảnh raw thực tế. Các camera `*_New` khác không được chọn. Không sửa JSON/raw.

Preflight split thật: 4 chunk train có 529 label, 1 chunk val có 171 label.
Có 10 frame train không đủ đồng bộ ảnh trong 40 ms (lệch lớn nhất khoảng 88 ms).
Dùng `--skip-unmatched` giữ 519 frame train và 171 frame val, ghi lý do bỏ frame
trong `conversion_report.json`. Không tăng ngưỡng đồng bộ mặc định.

## 1. Chọn profile nào?

| Mục đích | LiDAR-only | Fusion |
| --- | --- | --- |
| Sát batch của baseline trên từng GPU | `nuscenes/lidar.yaml` | `nuscenes/fusion.yaml` |
| Chạy trên GPU 8 GB hiện tại | `nuscenes/lidar_8gb.yaml` | `nuscenes/fusion_8gb.yaml` |
| Debug nhẹ, thông số cũ | `lidar.yaml` | `fusion.yaml` |

Các đường dẫn trên tương đối với `configs/pnkx/`. Torchpack nạp lần lượt
`configs/default.yaml` → `configs/pnkx/default.yaml` →
`configs/pnkx/nuscenes/default.yaml` → file được chọn. File `*_8gb.yaml` là leaf
cùng cấp với `fusion.yaml`/`lidar.yaml`, chọn nhánh bằng `use_camera`.

## 2. Những phần đã đưa về giống nuScenes

Baseline fusion:
[`swint_v0p075/convfuser.yaml`](../configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml).
Baseline LiDAR:
[`voxelnet_0p075.yaml`](../configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml).

| Thành phần | Profile mới | So với baseline |
| --- | --- | --- |
| Phạm vi point | `[-54,-54,-5,54,54,3]` m | Giống |
| Voxel | `[0.075,0.075,0.2]` m | Giống |
| Sparse shape / head grid | `[1440,1440,41]` | Giống |
| Max voxel train/test | `120000 / 160000` | Giống |
| LiDAR encoder | Hard voxelization + SparseEncoder, output 256 kênh | Giống |
| Camera backbone/neck | Swin-T + GeneralizedLSSFPN, 256 kênh | Giống |
| Ảnh đầu vào | 6 camera, mỗi ảnh `256×704` sau augmentation | Cùng số lượng/kích thước; sensor khác |
| Camera view transform | DepthLSSTransform, output 80 kênh | Giống |
| Bounds camera | x/y ±54, bước 0.3; depth `[1,60,0.5]`; downsample 2 | Giống |
| BEV cuối encoder | Camera `[B,80,180,180]`; LiDAR `[B,256,180,180]` | Giống |
| Fusion / decoder | ConvFuser 336→256; SECOND + SECONDFPN | Giống |
| Detection head | TransFusion, 200 proposals, hidden 128, 8 attention heads | Cấu trúc chung giống |
| Optimizer | AdamW, weight decay 0.01, clip gradient 35 | Giống |
| Fusion | LR `2e-4`, cosine, warmup linear 500 iteration, 6 epoch | Giống |
| LiDAR-only | LR `1e-4`, cyclic LR/momentum, 20 epoch | Giống |
| Precision | FP16, dynamic loss scaling, growth interval 2000 | Giống |
| Sampling | CBGSDataset | Giống phương pháp; tần suất theo 3 class PNKX |
| LiDAR augmentation | Scale `[0.9,1.1]`, quay ±π/4, translate 0.5, RandomFlip3D | Giống |
| Image augmentation train | Resize `[0.38,0.55]`, quay ±5.4°, random flip | Giống |
| GridMask fusion | Probability 0 | Giống leaf fusion baseline |

So với debug cũ, BEV tăng từ 128 lên 180, voxel XY giảm từ 0.1 xuống 0.075;
đây là thay đổi có ảnh hưởng đến kích thước feature, bộ nhớ và checkpoint.

### Khác biệt dành cho GPU 8 GB

| Tham số | Profile reference | Profile 8 GB |
| --- | --- | --- |
| Samples/GPU | 4 | 1 |
| Workers/GPU | 4 | 0 |
| Gradient accumulation | 1 | 4 |
| Swin activation checkpointing | Tắt | Bật |

Accumulation 4 giúp mỗi lần cập nhật dùng bốn micro-batch. BatchNorm vẫn chỉ
nhìn một sample mỗi lượt; kết quả không tương đương hoàn toàn batch 4. Baseline
README còn dùng 8 GPU, tức batch toàn hệ thống lớn hơn máy hiện tại. LR giữ theo
recipe tham chiếu, chưa được tối ưu riêng cho PNKX/1 GPU.

## 3. Những phần giữ khác và lý do

- **Class:** `car`, `rider`, `pedestrian`; heatmap có 3 đầu ra thay vì 10.
- **Velocity:** không có nhãn vx/vy nên bỏ nhánh velocity; bbox code size 8 thay
  vì 10 và bỏ hai velocity loss weights. Không thêm nhãn velocity bằng zero.
- **Point:** một sweep TOP, transform sensor→ego một lần; feature thứ năm là
  time lag 0. Chưa ghép sweep lịch sử khi chưa kiểm chứng ego pose/timestamp.
  Intensity giữ giá trị raw. Không giả định sensor có cùng phân bố với nuScenes.
- **Camera:** chọn sáu `CAM_P_*`, ảnh undistort giữ K; VF6_03 dùng nguồn
  `CAM_P_B_New` cho slot sau theo mapping nêu trên. Đọc ảnh, K, distortion và
  extrinsic cùng tên nguồn; không dùng calibration của camera cũ cho ảnh mới.
- **Dataset/evaluation:** PNKXDataset, split TXT theo recording, cache riêng;
  metric hiện là center-distance AP chẩn đoán, không phải nuScenes NDS/IoU AP.
- **Database sampling:** chưa tạo GT database PNKX nên không bật ObjectPaste.
  Pipeline baseline trong fork có ObjectPaste dù stop epoch làm nó không hoạt
  động ở fusion; vẫn không được trỏ sang database nuScenes.
- **Radar/GT depth:** không nạp radar hay target depth không dùng. Fork có radar
  loader trong pipeline mặc dù modality tắt; không sao chép loader đó sang PNKX.
- **Momentum fusion:** đặt `None` cho cosine recipe; tránh cyclic momentum còn
  sót do kế thừa config LiDAR trong fork. LiDAR vẫn dùng cyclic momentum.

Chạy `tools/compare_pnkx_configs.py` trong môi trường có torchpack để xuất config
đã resolve và bảng khác biệt tại `outputs/pnkx-config-review/`. Model reference
fusion chỉ còn 7 trường khác: đường dẫn pretrained camera và các trường gắn với
class/velocity/dataset. LiDAR còn 6 trường class/velocity/dataset. Bản fusion
8 GB thêm khác biệt `with_cp`.

## 4. Sửa quy ước box khi rà soát

Label/cache dùng `[x,y,z_center,length,width,height,yaw_CCW]`. Phiên bản box
legacy của repo yêu cầu `[x,y,z_bottom,width,length,height,yaw_internal]`, với
`yaw_internal = -yaw_CCW - π/2`.

[`pnkx_geometry.py`](../mmdet3d/datasets/pnkx_geometry.py) đổi kích thước/góc;
[`PNKXDataset.get_ann_info()`](../mmdet3d/datasets/pnkx_dataset.py) gọi hàm này rồi
constructor với `origin=(0.5,0.5,0.5)` để đổi tâm hình học sang tâm đáy đúng một lần.
Phép đổi giữ nguyên tám góc vật lý. Cache vẫn giữ raw convention nên không cần
convert lại cache chỉ vì sửa này. Hàm nghịch đảo yêu cầu geometric-center boxes;
không truyền trực tiếp tensor bottom-center vào hàm nghịch đảo.

**Adapter trước đây chỉ đổi origin, chưa đổi convention kích thước/góc.** Các
checkpoint debug cũ cần train lại; việc loss giảm trước đây chỉ chứng minh luồng
chạy, không chứng minh hình học box đúng. Test mới đối chiếu tám góc với công thức
raw độc lập và xác minh points/boxes cùng quay, scale, translate và flip.

## 5. Pretrained và cách chạy

Fusion mặc định dùng hai file local, theo quy trình pretrained trong README:

```text
pretrained/swint-nuimages-pretrained.pth
pretrained/lidar-only-det.pth
```

Nguồn tải nằm trong [`tools/download_pretrained.sh`](../tools/download_pretrained.sh)
và README. Script đó tải nhiều checkpoint, không bắt buộc chạy toàn bộ. Hai file
trên hiện **chưa có trong workspace**, chưa tự tải ở lần cấu hình này. Config
camera leaf nuScenes trỏ Swin ImageNet URL, nhưng quy trình train fusion trong
README override bằng Swin đã pretrain NuImages; profile mới theo quy trình đó.

`load_from` khởi tạo fusion từ LiDAR pretrained. Các layer head khác số class/
velocity không thể nạp nguyên vẹn; cần đọc log những key thiếu hoặc shape không
khớp, không coi đây là resume. Có thể ưu tiên checkpoint LiDAR-only PNKX đã train
với geometry/profile mới qua `--lidar_pretrained <path>`. LiDAR-only profile mặc
định train mới và không đòi hai file trên.

Sau khi có cache phù hợp, calibration đúng và pretrained:

```bash
bash tools/pnkx_docker.sh prepare
bash tools/pnkx_docker.sh check-fusion
bash tools/pnkx_docker.sh audit-fusion
bash tools/pnkx_docker.sh train-fusion
```

Với split VF6_03/VF6_01 hiện tại, convert toàn bộ raw bằng:

```bash
PYTHONPATH=/path/to/laz-dependencies python tools/data_converter/pnkx_converter.py \
  --root Data-PNKX --output data/pnkx \
  --train-split Data-PNKX/split/train.txt \
  --val-split Data-PNKX/split/val.txt \
  --with-cameras --camera-extrinsics ego-to-camera \
  --distortion-model opencv-standard --skip-unmatched
```

`/path/to/laz-dependencies` là môi trường có laspy/lazrs, NumPy và OpenCV; có thể
chạy converter trong Docker như hướng dẫn raw. Không bật `--allow-split-overlap`
và không giới hạn `--max-frames-per-chunk` cho split thật.

Các lệnh này chọn **fusion_8gb**, train mặc định **6 epoch**. LiDAR-only:

```bash
bash tools/pnkx_docker.sh check
bash tools/pnkx_docker.sh train
```

LiDAR train mặc định **20 epoch**. Thêm `-reference` để chọn batch 4, hoặc `-debug`
để chọn config debug. Ví dụ `train-fusion-reference`, `train-fusion-debug`.
Profile reference batch 4 chưa được xác minh bộ nhớ trên GPU 8 GB.

Đổi đường dẫn cache hoặc pretrained qua CLI:

```bash
bash tools/pnkx_docker.sh train-fusion \
  --dataset_root data/pnkx/ \
  --camera_pretrained pretrained/swint-nuimages-pretrained.pth \
  --lidar_pretrained runs/pnkx-lidar-nuscenes-8gb/epoch_20.pth \
  --run-dir runs/pnkx-fusion-nuscenes-next
```

Để thử pipeline ngắn khi chưa có pretrained, phải chủ động bỏ khởi tạo pretrained:

```bash
bash tools/pnkx_docker.sh audit-fusion --no-pretrained
bash tools/pnkx_docker.sh train-fusion \
  --camera_pretrained None --lidar_pretrained None \
  --fp16.loss_scale.init_scale 1.0 \
  --max_epochs 1 --run-dir runs/pnkx-fusion-random-smoke
```

CLI torchpack dùng Python literal `None`, không dùng chuỗi `null`. Override
`init_scale=1.0` dành cho smoke test từ random weights: scale mặc định của MMCV
là 65536, có thể overflow và bỏ những bước đầu để giảm scale. Không coi một
epoch chạy xong là đủ nếu mọi optimizer step đều bị bỏ. Kiểm tra `grad_norm`
hữu hạn và state optimizer thực sự có step. Checkpoint pretrained chưa được
kiểm chứng nạp trong lần này vì file chưa có.

## 6. Đã kiểm chứng trên máy hiện tại

- 12 test pass trong image `bevfusion:rtx5060`: dữ liệu/converter, biến đổi box,
  profile, khớp box/point sau augmentation và chọn đúng ảnh/intrinsic/extrinsic
  của camera sau khi train VF6_03 dùng New còn val VF6_01 dùng camera cũ.
- Cache split thật đã convert tại `data/pnkx`: 519 frame train trong 4 chunk,
  171 frame val trong 1 chunk; bỏ 10 frame train vượt ngưỡng timestamp, không bỏ
  chunk. Checker toàn cache và pipeline/collate train/val pass với 6 ảnh
  `256×704`. Audit GPU FP16 trên train VF6_03 pass, loss 184.3373 và cả ba nhánh
  có gradient hữu hạn, khác zero; xem
  [`pnkx-full-fusion-audit.json`](../outputs/pnkx-full-fusion-audit.json).
  Mapping nguồn và lý do bỏ frame lưu tại
  [`conversion_report.json`](../data/pnkx/conversion_report.json).
- Convert ba frame VF6_01 vào `/tmp/pnkx-profile-fixture/cache`, dùng split tạm
  trùng nhau có chủ đích; split thật giữ nguyên. CBGS tạo 9 lượt train từ 3 frame.
- Checker train/val pass với sáu ảnh và augmentation bật ở train.
- Audit CUDA FP16 pass, loss hữu hạn, point đầu vào không bị sửa tại chỗ;
  camera/LiDAR/fuser đều có gradient hữu hạn và khác zero. Peak allocated memory
  audit khoảng 2543 MiB. Report:
  [`pnkx-nuscenes-profile-audit.json`](../outputs/pnkx-nuscenes-profile-audit.json).
- LiDAR-only và fusion đều hoàn thành một epoch thử: 9 iteration train sau CBGS,
  validation 3 frame và lưu checkpoint. Với random weights, override initial
  loss scale 1.0 cho gradient hữu hạn và **3 optimizer step thật** mỗi mode;
  state optimizer được đọc lại từ checkpoint và kiểm tra hữu hạn. Fusion peak
  allocated memory trong log khoảng 2980 MiB, LiDAR khoảng 1568 MiB. Report:
  [`pnkx-nuscenes-training-smoke.json`](../outputs/pnkx-nuscenes-training-smoke.json).
  Các run tương ứng là `runs/pnkx-nuscenes-fixture-finite` và
  `runs/pnkx-nuscenes-lidar-fixture-finite`.

Lần thử fusion ban đầu giữ loss scale mặc định 65536: forward/validation chạy
nhưng gradient overflow ở cả ba lần dự định cập nhật. Không tính lần đó là
kiểm chứng optimizer thành công. Profile reference vẫn giữ dynamic scale của
baseline; override 1.0 nêu trên chỉ dùng cho phép thử random initialization.

Đây là kiểm tra luồng trên mẫu, chưa phải kết quả train toàn bộ PNKX hay đánh giá
chất lượng detection. Các run/log cũ vẫn giữ nguyên để tra cứu lịch sử.
