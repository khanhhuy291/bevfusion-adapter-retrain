# PNKX và nuScenes: đổi tên và đối chiếu cấu hình

> Bảng bên dưới là snapshot cấu hình debug tại lần đổi tên. Đã có
> [profile mới sát nuScenes](pnkx-nuscenes-profile.md), gồm grid 1440, BEV 180,
> CBGS, augmentation, FP16 và lịch train. Mapping camera sau VF6_03 từ
> `CAM_P_B_New` hiện đã được hỗ trợ; các hạn chế mapping nêu dưới là lịch sử.
> Dùng tài liệu mới để chọn cấu hình.

## Kết luận

Tên đúng hiện dùng thống nhất là **PNKX**. Các khối model chính của config
PNKX trùng baseline camera+LiDAR nuScenes trong repo, nhưng cấu hình hiện tại
vẫn là **bản debug**, chưa tương đương đầy đủ recipe nuScenes.

Lần này chỉ đổi tên và kiểm tra: không tự đổi grid, pretrained, batch, precision
hoặc lịch train. Đã đối chiếu snapshot trước/sau: ngoài tên PNXK → PNKX, các giá
trị config PNKX được giữ nguyên.

## 1. Phạm vi đổi tên

| Thành phần | Tên hiện tại |
| --- | --- |
| Raw root | `Data-PNKX/` — đã có tên đúng trước khi rà soát |
| Cache mặc định | `data/pnkx/` |
| Metadata | `pnkx_infos_train.pkl`, `pnkx_infos_val.pkl` |
| Config | `configs/pnkx/default.yaml`, `lidar.yaml`, `fusion.yaml` |
| Dataset / collect registry | `PNKXDataset`, `PNKXCollect3D` |
| Dataset tag của TransFusion | `PNKX` |
| Converter | `tools/data_converter/pnkx_converter.py` |
| Checker | `tools/check_pnkx.py`, `tools/check_pnkx_fusion.py` |
| Wrapper | `tools/pnkx_docker.sh` |
| Biến chọn image | `PNKX_IMAGE` |
| Tests / docs | Tên file và tham chiếu dùng `pnkx` |

Raw, nội dung split TXT và `.gitignore` do người dùng đã chỉnh được giữ nguyên.
Các log/checkpoint/report lịch sử trong `runs/pnxk-*`, `outputs/pnxk-*` giữ tên cũ;
link tới chúng trong docs vẫn trỏ đúng artifact gốc. Config dump trong run cũ là
snapshot của lần train đó, không được sửa hồi tố. Các lệnh mới dùng tên PNKX.

Thư mục `/tmp/pnxk-inspect-deps` trước đây đã bị dọn; dependency đọc LAZ cho
kiểm tra lần này được cài tại `/tmp/pnkx-inspect-deps`.

## 2. Baseline được chọn để so sánh

Fusion:

[`configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`](../configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml)

LiDAR-only:

[`configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml`](../configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml)

Đây là các baseline nuScenes **trong checkout hiện tại**, cũng là các đường dẫn
được README dùng. Không coi chúng là bản upstream nguyên trạng: pipeline ở repo
hiện tại có một số bước radar và thiết lập kế thừa cần lưu ý ở mục 6.

So sánh dùng config **đã resolve toàn bộ kế thừa và `${...}` bằng torchpack**,
không chỉ đọc riêng file leaf `fusion.yaml`.

Có thể xuất lại bằng image hiện có, từ gốc repo:

```bash
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$PWD:/work" -w /work -e PYTHONPATH=/work \
  bevfusion:rtx5060 tools/compare_pnkx_configs.py
```

Bằng chứng máy đọc được lưu trong `outputs/pnkx-config-review/`:

- `sources.json`: bốn config nguồn.
- `resolved-configs.json`: toàn bộ config đã resolve.
- `fusion-model-differences.json`: 23 field model khác nhau.
- `lidar-model-differences.json`: 19 field model khác nhau.
- `current-split-preflight.json`: kết quả kiểm tra calibration split hiện tại.

Số field khác không phải phần trăm khác biệt kiến trúc; một thay đổi voxel_size
có thể xuất hiện ở nhiều vị trí do interpolation của config.

## 3. Bảng so sánh kiến trúc fusion

| Thành phần | nuScenes baseline | PNKX hiện tại | Đánh giá |
| --- | --- | --- | --- |
| Model | BEVFusion | BEVFusion | Giống |
| Nhánh sensor | Camera + LiDAR, radar encoder không bật | Camera + LiDAR | Giống về loại nhánh |
| Camera input network | 256×704 | 256×704 | Giống |
| Camera backbone | SwinTransformer, embed 96 | SwinTransformer, embed 96 | Giống |
| Swin stages | Depths `[2,2,6,2]`, heads `[3,6,12,24]`, window 7 | Cùng giá trị | Giống |
| Swin output/dropout | Out indices `[1,2,3]`, drop path 0,2 | Cùng giá trị | Giống |
| Activation checkpointing | false | true | Khác cách giữ activation để giảm VRAM, không đổi số tầng/channel |
| Camera neck | GeneralizedLSSFPN, `[192,384,768]` → 256 | Cùng cấu hình neck | Giống |
| View transform | DepthLSSTransform, 256 → 80 channel | Cùng loại/channel | Giống |
| Depth bins | `[1,60,0.5]`: 118 bin | Cùng giá trị | Giống |
| Z bound / downsample | `[-10,10,20]` / 2 | Cùng giá trị | Giống |
| LiDAR voxelizer | Hard voxelization, max 10 point/voxel | Cùng loại/max point | Giống |
| LiDAR backbone | SparseEncoder, in 5, output_channels 128 | Cùng topology, channels, paddings và block_type | Giống phần mạng, khác grid |
| Fuser | ConvFuser `[80,256]` → 256 | Cùng cấu hình | Giống |
| Decoder backbone | SECOND, in 256, out `[128,256]`, layers `[5,5]` | Cùng cấu hình | Giống |
| Decoder neck | SECONDFPN `[128,256]` → `[256,256]` | Cùng cấu hình | Giống |
| Detection head | TransFusionHead, in 512, hidden 128 | Cùng loại/channel | Giống |
| Proposal / decoder / attention | 200 / 1 layer / 8 heads | Cùng giá trị | Giống |
| Assigner | HungarianAssigner3D với cùng chi phí class/BEV/IoU | Cùng toàn bộ config assigner | Giống |
| Loss | FocalLoss + GaussianFocalLoss + L1Loss | Cùng loại và trọng số loss | Giống các loss đang dùng |
| NMS | null | null | Giống |
| Map head | null | Không khai báo | Cả hai đều không train map head |

Phần architecture học feature và fuse đã rất gần. Tuy nhiên **không thể nói hai
config model giống hệt**, vì class, velocity và lưới không gian dưới đây khác nhau.

## 4. Grid, output và label

| Thuộc tính | nuScenes fusion | PNKX fusion | Lý do / hệ quả |
| --- | --- | --- | --- |
| Point range | `[-54,-54,-5,54,54,3]` | `[-51.2,-51.2,-5,51.2,51.2,3]` | Lựa chọn debug, không bắt buộc vì đổi dataset |
| Voxel size | `[0.075,0.075,0.2]` | `[0.1,0.1,0.2]` | PNKX thưa hơn, lưới nhỏ hơn |
| Sparse shape | `[1440,1440,41]` | `[1024,1024,41]` | Phải đồng bộ với range/voxel |
| Max voxels train/eval | `[120000,160000]` | `[90000,120000]` | Giới hạn tính toán khác |
| Camera XY bound | `[-54,54,0.3]` | `[-51.2,51.2,0.4]` | Camera grid phải khớp LiDAR sau downsample |
| Camera BEV trước downsample | 360×360 | 256×256 | Suy ra từ range / bước camera grid |
| BEV vào fuser/decoder | 180×180 | 128×128 | Khác độ phân giải feature dù channel giống nhau |
| Mét/ô BEV cuối | 0,6 | 0,8 | PNKX hiện coarser |
| Head grid config | `[1440,1440,41]` | `[1024,1024,40]` | Z physical grid là 40; sparse dùng padding 41. Head hiện dùng XY để tạo heatmap |
| Class | 10 class nuScenes | 3 class `car,rider,pedestrian` | Phải giữ đúng label gộp của PNKX |
| Velocity head | `[2,2]` | Bỏ | Label PNKX chưa có velocity |
| BBox code size | 10 | 8 | PNKX có geometry + sin/cos yaw, không vx/vy |
| Code weights | 8 số 1, thêm `[0.2,0.2]` cho velocity | 8 số 1 | Phần geometry giống, không supervision velocity giả |
| Small-object peak handling | Class nuScenes 8/9 | Rider/pedestrian PNKX 1/2 | Phải dùng đúng chỉ số class |
| Calibration / frame | NuScenes metadata và sensor calibration | Point cache ego, ma trận thực của xe PNKX | Không thể dùng calibration tác giả cho xe khác |

Muốn sát baseline v0p075 hơn về **grid**, phải đổi đồng bộ range, voxel size,
sparse shape, head train/test grid, coder voxel/range và camera x/y bound. Chỉ
đổi một dòng voxel_size sẽ làm các phần không khớp. Chưa thực hiện thay đổi này
và chưa kiểm tra VRAM cho profile 180×180 trên máy 8 GB.

## 5. Pipeline và recipe train

| Thành phần | nuScenes fusion đã resolve | PNKX hiện tại |
| --- | --- | --- |
| Dataset / sampler | NuScenesDataset trong CBGSDataset | PNKXDataset trực tiếp, chưa CBGS |
| Class source | NuScenes annotations | TXT custom qua converter |
| Point features | 5 chiều | 5 chiều: xyz ego, raw intensity, zero sweep_dt |
| Sweep loader train | Có LoadPointsFromMultiSweeps, sweeps_num=0 | Một sweep TOP, không loader lịch sử |
| Sweep loader eval | sweeps_num=9 | Một sweep TOP |
| reduce_beams | 32 | null; loader chỉ giảm nếu nhỏ hơn 32, nên cả hai cấu hình này không giảm beam |
| Image augmentation train | Resize `[0.38,0.55]`, rotate ±5,4°, random flip | Resize cố định 0,48, không rotate/flip |
| Image augmentation eval | Resize 0,48, không rotate/flip | Cùng tham số số học; raw aspect ratio khác nên vùng crop thực khác |
| 3D augmentation | Scale `[0.9,1.1]`, yaw ±π/4, translate 0,5, RandomFlip3D | Identity; tắt augmentation hình học để debug |
| Image normalization | Mean/std ImageNet | Cùng mean/std |
| GridMask | Có bước, prob=0 | Bỏ bước; ở prob=0 không tạo khác biệt augmentation |
| ObjectPaste | Có bước; fusion stop_epoch=-1 | Bỏ bước, chưa tạo GT database PNKX |
| Radar loader / collection | Vẫn tồn tại trong pipeline của checkout, dù modality.use_radar=false | Không có radar loader/field |
| GTDepth pipeline | Có bước | Không chạy GTDepth; DepthLSS tự dùng sparse depth từ input point |
| Optimizer | AdamW, LR 2e-4, weight_decay 0,01 | AdamW, LR 1e-4, weight_decay 0,01 |
| Gradient clipping | max_norm 35 | Cùng giá trị |
| LR scheduler | CosineAnnealing + linear warmup 500 iter, ratio 1/3, min ratio 1e-3 | CosineAnnealing, không warmup, min ratio 1e-3 |
| Momentum config | Kế thừa `policy: cyclic` trong checkout | null |
| Epoch trong config | 6 | 2; wrapper debug override 1 |
| Batch / workers mỗi GPU | 4 / 4 | 1 / 0 |
| Precision | fp16 loss scaling | FP32 |
| Logging | Mỗi 50 iter, text + TensorBoard | Mỗi 10 iter, text |
| Eval metric | NuScenes evaluator (mAP/NDS và metric liên quan) | AP diagnostic theo khoảng cách tâm, không IoU AP/NDS |

Pretrained cũng khác và cần phân biệt config với lệnh README:

- NuScenes leaf config có Swin init_cfg trỏ checkpoint pretrained ImageNet.
- Lệnh fusion trong README override camera checkpoint thành Swin NuImages,
  đồng thời `--load_from pretrained/lidar-only-det.pth`.
- PNKX hiện `init_cfg=null`, `load_from=null`: train từ khởi tạo mới, không
  tự sử dụng checkpoint LiDAR-only đã train trong phiên debug.

Vì vậy cùng topology **không đồng nghĩa** đã tái lập recipe hoặc chất lượng baseline.

### LiDAR-only

PNKX LiDAR-only cũng giữ SparseEncoder → SECOND → SECONDFPN → TransFusionHead
như baseline LiDAR-only v0p075. Các khác biệt grid/class/velocity/batch/FP32 ở trên
vẫn áp dụng. Riêng optimizer/schedule:

| Thuộc tính | nuScenes LiDAR-only | PNKX LiDAR-only |
| --- | --- | --- |
| LR | 1e-4 | 1e-4 |
| LR / momentum scheduler | Cyclic / cyclic | CosineAnnealing / null |
| Epoch config | 20 | 2; wrapper 1 |
| Database paste stop | 15 | Không paste |

## 6. Các điểm không nên sao chép máy móc từ checkout nuScenes

1. **Radar pipeline còn sót:** LoadRadarPointsMultiSweeps đọc `results['radar']`
   trực tiếp. Modality flag false không tự làm transform này biến mất. Không
   thêm radar vào PNKX chỉ để danh sách pipeline giống nhau.
2. **Train 0 sweep / eval 9 sweep:** đây là giá trị thực sau kế thừa trong checkout,
   không nên mô tả chung rằng baseline luôn dùng 9 sweep cả train lẫn eval.
3. **ObjectPaste stop=-1:** với `dataset.set_epoch(0)`, điều kiện epoch>=stop bỏ
   paste trong fusion, nhưng sampler vẫn được khởi tạo và có thể cần dbinfos.
   Không cần dựng database PNKX chỉ để giữ một bước không paste ở recipe này.
4. **GTDepth không đồng nghĩa có depth loss:** DepthLSSTransform hiện có
   `use_depth_loss=false`; không supervision bằng GT box depth ở PNKX.
5. **Shared implementation:** các sửa projection/depth/BEV boundary nằm trong code
   dùng chung. So sánh này không khẳng định source hiện tại giống bit-for-bit với
   upstream hoặc paper ban đầu.

## 7. Split mới hiện có một vấn đề sensor cần xử lý trước convert fusion

Train TXT hiện trỏ hai recording VF6_03 tháng 7; val trỏ recording VF6_01 mẫu cũ.
Cả hai recording train có hai chunk. Raw root và recording name resolve đúng.

Tuy nhiên bộ camera cố định hiện tại yêu cầu `CAM_P_B`, trong khi VF6_03 có
`CAM_P_B_New` trong cả thư mục, intrinsics và extrinsics. VF6_03 cũng có nhiều
camera `_New` khác. Kiểm tra calibration hiện báo thiếu `CAM_P_B` cho hai recording
train. Val VF6_01 có đủ key calibration của bộ sáu camera hiện tại.

Chưa tự thay `CAM_P_B_New` vào `CAM_P_B`: cần xác nhận bộ camera theo xe, góc nhìn
và projection; tên gần giống không đủ để kết luận tương đương hình học. Đây là
khác biệt sensor của dữ liệu mới, không phải lỗi do đổi tên PNKX.

Cache mặc định `data/pnkx/` hiện chưa tồn tại trong workspace. Vì vậy lần này kiểm
tra registry/config/tests/preflight, không tuyên bố đã nạp batch hoặc train lại
trên split mới. Raw và split hiện tại không bị chỉnh để làm test vượt qua.

## 8. Muốn sát nuScenes hơn thì ưu tiên gì?

Các mục bắt buộc giữ theo dữ liệu: ba class, không velocity khi không có GT,
calibration riêng, dataset/split riêng và undistort camera đúng mô hình.

Các lựa chọn debug có thể thay sau khi mapping VF6_03 được xác minh:

1. Pretrained camera/LiDAR tương thích, kiểm tra các head/tensor shape khác.
2. Grid v0p075/180×180 đồng bộ nếu phù hợp VRAM và phạm vi mục tiêu.
3. Image/3D augmentation thay cho identity; kiểm tra lại projection khi augment.
4. LR/warmup/epoch theo recipe, điều chỉnh theo batch và số GPU thực tế.
5. CBGS, multi-sweep hoặc database sampling chỉ khi đã có dữ liệu/metadata và
   mục đích phù hợp; multi-sweep cần pose/bù chuyển động, không chỉ bật một số.
6. Evaluator IoU 3D phù hợp PNKX; không gọi diagnostic AP hiện tại là NDS.

## 9. Kiểm tra sau đổi tên

- Không còn tham chiếu PNXK trong source/config/test đang chạy.
- Registry `PNKXDataset` và `PNKXCollect3D` import thành công trong image hiện có.
- Cả bốn config resolve qua torchpack; topology chung và các khác biệt trên được
  xác nhận từ JSON đã resolve.
- So sánh trước/sau rename xác nhận config PNKX không bị đổi hyperparameter.
- Bộ test converter/metric/depth CPU, compile Python và cú pháp shell được chạy lại.
- `.gitignore` hiện loại `Data-PNKX/*`; nội dung chỉnh sửa này của người dùng được giữ.
