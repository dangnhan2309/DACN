import os

folder = r"D:\yolo\datasets\images"

files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Sắp xếp để rename theo thứ tự
files.sort()

# Đổi tên lần lượt
for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]  # lấy phần mở rộng (.jpg, .png,...)
    new_name = f"img{i:04d}{ext}"       # img0001.jpg, img0002.jpg ...
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)

print("✅ Done! Đã rename toàn bộ ảnh.")
