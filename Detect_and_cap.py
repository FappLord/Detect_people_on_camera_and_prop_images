import cv2
import os
import time

# --- CÁC THAM SỐ CẤU HÌNH ---
CAMERA_INDEX = 0
OUTPUT_FOLDER = "auto_captured_images"

# Đường dẫn đến file XML của Haar Cascade
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"


# Khoảng thời gian nghỉ (cooldown) giữa các lần chụp (tính bằng giây)
# Hệ thống sẽ chỉ chụp 1 ảnh mỗi 1 giây nếu vẫn thấy khuôn mặt
CAPTURE_COOLDOWN = 1  # 1 giây

def auto_capture_on_detect():
    """
    Tự động chụp ảnh khi phát hiện khuôn mặt, với khoảng thời gian nghỉ giữa các lần chụp.
    """
    # 1. Tải bộ phân loại khuôn mặt Haar Cascade
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Lỗi: Không tìm thấy file '{HAAR_CASCADE_PATH}'.")
        print("Vui lòng tải file và đặt nó vào cùng thư mục với script.")
        return
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 2. Khởi động webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Lỗi: Không thể khởi động webcam {CAMERA_INDEX}.")
        return

    print("\nHệ thống đã sẵn sàng. Vui lòng đứng trước camera.")
    print("Nhấn 'q' để thoát.")
    
    window_name = "Automatic Attendance - Press 'q' to quit"
    last_capture_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1) # Lật ảnh cho tự nhiên
        
        # Tạo một bản sao của khung hình để vẽ lên, giữ lại ảnh gốc để lưu
        display_frame = frame.copy()
        
        # Chuyển sang ảnh xám để tăng tốc độ phát hiện
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt trong ảnh xám
        # scaleFactor và minNeighbors là các tham số tinh chỉnh, có thể thay đổi
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Kiểm tra xem có cần chụp ảnh hay không
        can_capture = (time.time() - last_capture_time) > CAPTURE_COOLDOWN

        # Vẽ hình chữ nhật xung quanh các khuôn mặt được phát hiện
        for (x, y, w, h) in faces:
            color = (0, 255, 0) # Xanh lá cây
            # Nếu có thể chụp, vẽ hình chữ nhật màu xanh
            if can_capture:
                cv2.putText(display_frame, "Ready to Capture", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

        # Nếu có khuôn mặt và đã hết thời gian nghỉ -> Chụp ảnh
        if len(faces) > 0 and can_capture:
            # Lấy thời gian hiện tại để đặt tên file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"capture_{timestamp}.jpg"
            image_path = os.path.join(OUTPUT_FOLDER, image_name)
            
            # Lưu khung hình gốc (không có hình vẽ)
            cv2.imwrite(image_path, frame)
            
            print(f"ĐÃ LƯU: {image_path}")
            
            # Cập nhật lại thời gian chụp cuối cùng
            last_capture_time = time.time()

            # Hiển thị thông báo "SAVED!" trên màn hình
            cv2.putText(display_frame, "SAVED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        elif not can_capture:
            # Hiển thị thời gian cooldown còn lại
            remaining_time = CAPTURE_COOLDOWN - (time.time() - last_capture_time)
            cv2.putText(display_frame, f"Cooldown: {remaining_time:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị video lên màn hình
        cv2.imshow(window_name, display_frame)

        # Thoát bằng phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nChương trình kết thúc.")

# --- Chạy chương trình ---
if __name__ == "__main__":
    auto_capture_on_detect()