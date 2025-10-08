from roboflow import Roboflow
import json
import cv2

# ✅ Khởi tạo kết nối tới Roboflow
rf = Roboflow(api_key="wxmpb1PyA5I08QLpxyH5")
project = rf.workspace("nguyen-utdt0").project("my-first-project-4rdca")
model = project.version(2).model

# ✅ Đường dẫn ảnh test
image_path = r"D:\yolo\datasets\images\img0090.jpg"

# ✅ Dự đoán (predict) ảnh và lưu ảnh có bounding box
result = model.predict(image_path, confidence=40, overlap=30)
result.save(r"D:\yolo\datasets\results\result_img0090.jpg")

# ✅ Lấy dữ liệu JSON (thông tin chi tiết các đối tượng phát hiện)
prediction = result.json()

print("===== 📊 KẾT QUẢ PHÁT HIỆN =====\n")

# In chi tiết từng đối tượng phát hiện
if "predictions" in prediction and len(prediction["predictions"]) > 0:
    for i, obj in enumerate(prediction["predictions"], 1):
        print(f"🔹 Đối tượng #{i}")
        print(f"  - Nhãn (Label): {obj['class']}")
        print(f"  - Độ tin cậy (Confidence): {obj['confidence']*100:.2f}%")
        print(f"  - Tọa độ khung (x,y,w,h): ({obj['x']:.1f}, {obj['y']:.1f}, {obj['width']:.1f}, {obj['height']:.1f})")
        print("-" * 50)
else:
    print("⚠️ Không phát hiện được đối tượng nào trong ảnh.")

# ✅ Hiển thị ảnh có khung kết quả
result_image_path = r"D:\yolo\datasets\images\result_img0090.jpg"
img = cv2.imread(result_image_path)
if img is not None:
    cv2.imshow("Kết quả phát hiện", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"\n✅ Ảnh kết quả đã lưu tại: {result_image_path}")
