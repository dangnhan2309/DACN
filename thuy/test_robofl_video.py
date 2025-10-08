import cv2
import os
from roboflow import roboflow

rf = Roboflow(api_key="wxmpb1PyA5I08QLpxyH5")
project = rf.workspace("nguyen-utdt0").project("my-first-project-4rdca")
model = project.version(2).model  


input_video_path = "parkour2.mp4" # Video gốc cần xử lý
output_video_path = "datasets/results/result_parkour2.mp4" # Video kết quả đầy đủ, có vẽ bounding box.
output_danger_path = "datasets/results/result_danger_parkour2.mp4" # Video chỉ chứa các frame có hành vi nguy hiểm

cap = cv2.VideoCapture(input_video_path) # Mở video để đọc frame.
if not cap.isOpened(): # Kiểm tra video có mở được không
    raise Exception(f"Không thể mở video: {input_video_path}")


# cap.get() trả về giá trị kiểu float.
# Nhưng khi tạo video mới với VideoWriter, kích thước phải là số nguyên (int)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# OpenCV trả về FPS mà video gốc được ghi.
# fps là số khung hình trên giây (Frames Per Second) của video
fps    = cap.get(cv2.CAP_PROP_FPS)
# fourcc là viết tắt của “Four Character Code”
# Đây là Codec là phần mềm/thuật toán nén dữ liệu video để
# Dấu * trong Python unpack chuỗi thành 4 đối số riêng lẻ: 'm','p','4','v'.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
out_danger = cv2.VideoWriter(output_danger_path, fourcc, fps, (width, height))

print("🚀 Bắt đầu xử lý video... (nhấn Q để thoát)")

# ================= 3. Biến đếm =================
danger_count = 0
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    # confidence=0.5: chỉ lấy dự đoán ≥ 50% tin cậy.
    # overlap=30: kiểm soát bounding box bị trùng lặp.
    results = model.predict(frame, confidence=0.5, overlap=30).json()  
    predictions = results.get('predictions', [])

    danger_in_frame = False

    for pred in predictions:
        x1 = int(pred['x'] - pred['width']/2)
        y1 = int(pred['y'] - pred['height']/2)
        x2 = int(pred['x'] + pred['width']/2)
        y2 = int(pred['y'] + pred['height']/2)
        cls = pred['class']        # 'normal' hoặc 'suicide'
        conf = pred['confidence']

        # Vẽ bounding box
        if cls.lower() == 'suicide_attempt':
            color = (0,0,255)  # đỏ
            danger_in_frame = True
            print(f"[Frame {frame_index}] ⚠️ Cảnh báo: phát hiện hành vi nguy hiểm với confidence {conf:.2f}")
        else:
            color = (0,255,0)  # xanh

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Lưu video đầy đủ
    out.write(frame)

    # Nếu frame có hành vi nguy hiểm, lưu vào video riêng
    if danger_in_frame:
        out_danger.write(frame)
        danger_count += 1

    # Hiển thị realtime (tùy chọn)
    cv2.imshow('Result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
out_danger.release()
cv2.destroyAllWindows()

print(f"✅ Video kết quả đã lưu tại: {output_video_path}")
print(f"✅ Video các frame nguy hiểm đã lưu tại: {output_danger_path}")
print(f"⚠️ Tổng số frame nguy hiểm: {danger_count}")
