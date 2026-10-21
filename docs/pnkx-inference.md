# Inference và vẽ GT/prediction cho PNKX

Chạy từ thư mục repo, dùng cache `data/pnkx` và checkpoint đã train. Không cần
convert lại raw. Tool dùng pipeline **val**, model **eval**, một GPU, FP16 theo
config của lần train; không khởi tạo MPI và không tải pretrained.

## Lệnh chạy

```bash
cd /home/khanhhuy/bevfusion-adapter-retrain

bash tools/pnkx_docker.sh visualize-fusion \
  --config runs/pnkx-fusion-full/configs.yaml \
  --checkpoint runs/pnkx-fusion-full/epoch_6.pth \
  --frames 0 40 80 120 170 \
  --classes pedestrian \
  --score-threshold 0.1 \
  --out-dir outputs/pnkx-val-pedestrian
```

`--frames` là index zero-based của dataset val. Bỏ cờ này để chọn tự động 5 frame
cách đều; đổi số lượng bằng `--num-frames`. Config mặc định và checkpoint mặc
định chính là hai file trong lệnh trên. Có thể override cache bằng
`--dataset-root data/pnkx/`.

Để xem thêm dự đoán score thấp, đổi thành `--score-threshold 0.03` và chọn
output khác, ví dụ `--out-dir outputs/pnkx-val-pedestrian-score003`. Ngưỡng thấp
hiện nhiều dự đoán hơn, gồm cả false positive; không tự cải thiện chất lượng.

Để vẽ cả ba class:

```bash
bash tools/pnkx_docker.sh visualize-fusion \
  --frames 0 40 80 120 170 \
  --classes car rider pedestrian \
  --score-threshold 0.1 \
  --out-dir outputs/pnkx-val-all-classes
```

## Đầu ra và cách đọc

Mỗi frame có một thư mục `index_timestamp/`, chứa:

- `bev.png`: bên trái là toàn BEV, bên phải phóng to quanh GT gần ego nhất trong
  nhóm class đã chọn. Đổi bán kính phóng to bằng `--zoom-radius 10` (mét).
- `cameras.jpg`: sáu ảnh cache đã undistort, chồng GT và prediction. Hàng trên
  theo thứ tự F, FL, FR; hàng dưới B, LB, RB. `--camera-width 960` chọn chiều
  rộng từng ảnh trong montage.
- `boxes.json`: toàn bộ GT và prediction, score, class, ID box, danh sách box
  được vẽ, cặp ghép theo tâm XY trong 2 m và sai số tâm/yaw của những cặp đó.

`summary.json` ở thư mục output tổng hợp số box được vẽ và số cặp ghép từng
frame. Đây là chẩn đoán trên vài frame có lọc score, không phải AP toàn val.

Màu **xanh lá = GT**, **đỏ = prediction**. Mũi tên chỉ hướng yaw, số `deg` trên
BEV zoom là yaw trong ego tính bằng độ. ID `g...` và `p...` liên kết với JSON;
số thập phân sau ID prediction là confidence. Trên BEV, ego ở `(0,0)`, phía
trước hướng lên, phía trái xe nằm bên trái hình. Mũi tên dùng chiều dài tối
thiểu để dễ nhìn, còn đường viền box giữ kích thước vật lý thật.

## Quy ước hình học và checkpoint

Tool [`tools/visualize_pnkx.py`](../tools/visualize_pnkx.py) nạp toàn bộ state
checkpoint với `strict=True` để kiểm tra đủ key/shape. Thông báo constructor
Swin "No pre-trained weights" có thể xuất hiện trước khi nạp; sau đó phải có
`Loaded full trained checkpoint ... | evaluation only`. Đó là inference từ
checkpoint hoàn chỉnh, không phải train mới.

Box prediction của framework dùng tâm đáy và quy ước WLH/yaw legacy. Tool lấy
`gravity_center` đúng một lần rồi gọi `internal_boxes_to_raw()` để đưa về
`[x,y,z_center,length,width,height,yaw_CCW]`. GT trong cache vốn đã ở quy ước
này. Không trừ thêm height/2 hay coi tensor nội bộ là LWH raw.

Projection vẽ trên ảnh cache ở kích thước gốc bằng intrinsic và ego→camera của
chính sensor đó. Ảnh đưa vào model vẫn đi qua resize/crop của pipeline val và
`img_aug_matrix`; không dùng trực tiếp ảnh đã normalize để vẽ. Tool kiểm tra
`lidar_aug_matrix=I` trước khi vẽ và clip các cạnh box tại camera depth 0.1 m.

Sai số yaw được wrap theo vòng 360°. Với box gần vuông hoặc vật thể không rõ
hướng, cần đọc cùng hình và nhãn; giá trị này không phải metric nuScenes AOE.
