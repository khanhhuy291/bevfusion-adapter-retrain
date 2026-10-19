# Quy trình kiểm chứng dữ liệu PNXK trước khi adapt BEVFusion

Ngày thực hiện: 19/10/2026.

## 1. Mục tiêu và phạm vi

Kiểm tra khả năng đọc LAZ, quy ước box, đồng bộ timestamp và chiều extrinsics
trước khi viết converter/dataset cho BEVFusion. Chưa thay đổi pipeline train.

Dữ liệu khảo sát nằm trong `Data-PNXK`, với recording
`20260228_1105_VF6_01_1772251511_1772254880`, chunk
`37_1772252610_1772252644`.

Chunk có 171 frame label. Chọn ba frame đầu, giữa và cuối theo timestamp:

| Frame | Số điểm LIDAR_TOP | Số box | Box có điểm TOP sau khi tính lại |
| --- | ---: | ---: | ---: |
| `1772252610-099740267` | 216.495 | 105 | 105 |
| `1772252627-099736452` | 219.548 | 157 | 121 |
| `1772252643-999713421` | 212.991 | 119 | 119 |

## 2. Quy ước được chủ dữ liệu xác nhận

- Ego: x hướng trước, y hướng trái, z hướng lên; đơn vị mét.
- Label: `x y z length width height yaw class occlusion_level number_of_point track_id`.
- `x,y,z` là tâm hình học của box.
- Yaw tính bằng radian, bằng 0 khi chiều dài hướng +x; yaw dương quay từ +x sang +y.
- Point trong LAZ nằm ở hệ tọa độ sensor.
- Ba class: `Car`, `Rider`, `Pedestrian`. Car gộp car/truck/trailer/bus/construction_vehicle;
  Rider gộp bicycle/motorcycle.
- Ảnh chưa xử lý méo.

Ban đầu extrinsics được mô tả chung là sensor → ego. Kiểm tra thực tế dưới đây
cho thấy cần phân biệt chiều ma trận camera với LiDAR trong calibration mẫu.

## 3. Kiểm tra cấu trúc, label và split

Đọc `split/train.txt`, `split/val.txt`; resolve recording rồi duyệt các chunk bên trong.
Hai split hiện cùng recording để kiểm tra pipeline. Không dùng metric từ split trùng
để đánh giá khả năng tổng quát hóa; khi train thật cần kiểm tra overlap theo recording.

Đã kiểm tra toàn bộ label trong chunk:

- 26.653 box, tất cả dòng có 11 trường.
- Chỉ có ba class đã mô tả; kích thước box đều dương.
- Yaw xấp xỉ trong khoảng `[-π, π]`.
- 4.674 box có `number_of_point = 0`.
- Có box ngoài phạm vi ±100 m: phạm vi mặc định ±51,2 m của config sẽ loại một phần box.

Không tự động coi box không có điểm là annotation sai. Chính sách lọc cần được
chọn theo sensor thực sự sử dụng và mục tiêu LiDAR-only hay fusion.

## 4. Ghép timestamp

Tên file có dạng `seconds-nanoseconds.ext`. Parse bằng số nguyên:

```python
timestamp_ns = int(seconds) * 1_000_000_000 + int(nanoseconds)
```

Lấy timestamp label làm mốc, tìm file sensor có độ lệch tuyệt đối nhỏ nhất.
Không ghép theo vị trí trong danh sách hoặc yêu cầu tên file camera giống tên label.

Kết quả trên toàn bộ 171 label:

| Sensor | Độ lệch timestamp gần nhất |
| --- | --- |
| LIDAR_TOP | 171/171 khớp tuyệt đối |
| Bốn LiDAR phụ | Khoảng 3,7–4,1 ms |
| Camera, trừ CAM_P_LB/RB | Tối đa khoảng 2,1 ms |
| CAM_P_LB | Tối đa khoảng 32,5 ms |
| CAM_P_RB | Tối đa khoảng 33,1 ms |

Script kiểm chứng chọn nearest timestamp và lưu độ lệch, chưa áp dụng ngưỡng loại
frame. Converter train cần ngưỡng rõ ràng và báo cáo trường hợp vượt ngưỡng.
Khớp timestamp theo tên file chưa chứng minh các sensor đã được đồng bộ vật lý
hoặc đã bù chuyển động trong một vòng quét.

## 5. Đọc LAZ và đưa point về ego

Dùng `laspy` với backend `lazrs`; đọc thành công file mẫu của cả năm LiDAR.
Các file có XYZ, intensity và extra dimensions `ring`, `point_timestamp`.
Chỉ LIDAR_TOP được dùng để kiểm chứng hình học/projection trong bước này.

Đọc `las.x`, `las.y`, `las.z` để nhận tọa độ đã áp dụng scale/offset của LAS,
không sử dụng trực tiếp các trường số nguyên `X`, `Y`, `Z` như mét.

Chọn calibration từ `VF6_01_Extrinsics_By_Dates.json`, khoảng
`2026-02-28 <= ngày recording < 2026-03-20`.

Quy ước ký hiệu `T_A_from_B` biến đổi điểm từ B sang A, dùng vector cột:

```text
p_ego = T_ego_from_lidar × p_lidar
```

Với mảng NumPy lưu mỗi điểm trên một hàng:

```python
points_ego = points_sensor @ T[:3, :3].T + T[:3, 3]
```

Ma trận LIDAR_TOP trong JSON được dùng trực tiếp theo chiều sensor → ego.

## 6. Kiểm chứng box bằng BEV và đếm điểm

Tạo tám góc box quanh tâm từ `±length/2`, `±width/2`, `±height/2`, quay quanh z
theo yaw rồi cộng tâm ego. Vẽ point và box ở BEV: +x lên trên, +y sang trái;
đường trắng từ tâm chỉ hướng yaw.

Đếm điểm trong box bằng cách đưa point ego về hệ trục box:

```text
p_local = Rz(yaw)^T × (p_ego - center)
inside = |p_local| <= [length/2, width/2, height/2]
```

Box và cloud TOP khớp hợp lý qua quan sát. Trong 381 box của ba frame,
244 box có số điểm tính lại bằng chính xác trường `number_of_point`;
trung vị sai khác tuyệt đối bằng 0 ở từng frame. Frame giữa có 36 box không
có điểm TOP, trùng số box được ghi zero-point trong label.

Không phải mọi count đều giống nhau: chưa kết luận quy trình tạo trường
`number_of_point` và chưa xác nhận label được tạo từ cloud nào.

## 7. Thử hai chiều extrinsics camera

Kiểm tra sáu camera: `CAM_P_F`, `CAM_P_FL`, `CAM_P_FR`, `CAM_P_B`,
`CAM_P_LB`, `CAM_P_RB` trên cả ba frame.

Với ma trận camera `M` lấy từ JSON, dựng hai giả thuyết trên cùng ảnh:

- A, bên trái: `p_camera = M × p_ego`.
- B, bên phải: `p_camera = inverse(M) × p_ego`.

Để giữ cùng hình học cho ảnh và phép chiếu, ảnh được undistort thử bằng
`cv2.undistort(image, K, distortion, None, K)`, giữ nguyên K. Đây là **giả định
mô hình distortion chuẩn OpenCV cho CAM_P**, chưa phải xác nhận lens model.
CAM_F fisheye chưa được kiểm tra.

Chiếu point và góc box bằng `K`, chia theo depth camera. Chỉ vẽ point có depth
trên 0,5 m; chỉ vẽ cạnh box khi cả hai đầu có depth trên 0,5 m. Ảnh được thu
nhỏ từ 1920×1536 xuống 640×512 nên tọa độ projection cũng chia 3.
Point dùng màu theo depth; box Car màu cam, Rider xanh lá, Pedestrian hồng.

Kết quả quan sát: giả thuyết A khớp cảnh rõ ràng hơn; B gây lệch mạnh và box
xiên sai hướng. Kết luận áp dụng cho sáu camera và calibration mẫu:

```text
T_camera_from_ego = M_camera_JSON
T_ego_from_camera = inverse(M_camera_JSON)

T_camera_from_lidar = T_camera_from_ego × T_ego_from_lidar
```

Không áp dụng chung quy tắc “mọi extrinsics JSON đều sensor → ego”.
Ảnh vẫn có điểm/box của vật thể bị che vì chưa lọc occlusion theo depth ảnh;
điều này tự nó không chứng minh calibration sai.

## 8. Chạy lại và xem kết quả

Script kiểm chứng đã lưu tại
[`outputs/pnxk-inspection/inspect_pnxk.py`](../outputs/pnxk-inspection/inspect_pnxk.py).
Script hiện cố định root workspace, chunk mẫu và calibration period đầu tiên;
cần cập nhật khi chuyển máy hoặc kiểm tra recording khác.

Dependency kiểm chứng được cài riêng vào `/tmp`, không thay đổi môi trường train:

```bash
python -m pip install --target /tmp/pnxk-inspect-deps --no-deps laspy lazrs opencv-python-headless
mkdir -p outputs/pnxk-inspection
python outputs/pnxk-inspection/inspect_pnxk.py
```

Python đang dùng cần có NumPy và Pillow. `/tmp` có thể bị dọn, khi đó cần cài lại
dependency. Cài `--no-deps` dựa trên NumPy sẵn có trong môi trường đã kiểm tra.

Artifact đầu ra:

- `*_bev.jpg`: box và point trong ego, nhìn từ trên xuống.
- `*_CAM_*_compare.jpg`: A bên trái, B bên phải.
- `*_all_cameras.jpg`: sáu hàng F, FL, FR, B, LB, RB.
- `inspection.json`: timestamp ảnh, độ lệch và số điểm từng box.

Ví dụ:

- [Camera trước, frame đầu](../outputs/pnxk-inspection/1772252610-099740267_CAM_P_F_compare.jpg).
- [Sáu camera, frame giữa](../outputs/pnxk-inspection/1772252627-099736452_all_cameras.jpg).
- [BEV, frame đầu](../outputs/pnxk-inspection/1772252610-099740267_bev.jpg).

`outputs/*` và `Data-PNXK/*` đang được gitignore. Tài liệu này được lưu trong docs;
script/ảnh dẫn tới ở trên là artifact cục bộ, không tự đi kèm khi clone repo.

## 9. Giới hạn và hệ quả cho adapter

Đây là kiểm chứng trực quan trên một chunk, không có metric reprojection định lượng.
Chưa kiểm chứng mọi xe, mọi khoảng ngày, camera fisheye, alignment giữa năm LiDAR,
deskew, motion compensation hoặc độ chính xác lens model.

Adapter cho mẫu hiện tại có thể dựa trên box ego và point TOP đã transform về ego;
camera dùng chiều biến đổi đã xác minh ở mục 7. Khi mở rộng dữ liệu cần chọn
calibration theo xe/ngày, kiểm tra đủ sensor, đồng bộ intrinsics với ảnh undistort
và kiểm chứng lại projection trên các cấu hình mới.

Khả năng đọc dữ liệu không đồng nghĩa môi trường train sẵn sàng: lúc khảo sát,
`nvidia-smi` không giao tiếp được với driver; Python/PyTorch hiện tại khác stack
cũ repo yêu cầu và chưa có mmcv/mmdet. Chưa chạy train trong quy trình này.
