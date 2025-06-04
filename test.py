import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  # Ganti ke 1 jika pakai USB camera

    if not cap.isOpened():
        print("Kamera gagal dibuka")
        return

    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA tersedia: {use_cuda}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        if use_cuda:
            # Proses di GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            hsv_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
            upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
            lower_red2 = np.array([170, 120, 70], dtype=np.uint8)
            upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

            mask1 = cv2.cuda.inRange(hsv_gpu, lower_red1, upper_red1)
            mask2 = cv2.cuda.inRange(hsv_gpu, lower_red2, upper_red2)
            red_mask_gpu = cv2.cuda.bitwise_or(mask1, mask2)

            # Unduh mask ke CPU
            red_mask = red_mask_gpu.download()
        else:
            # Proses di CPU
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)

        # Temukan kontur
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Abaikan noise kecil
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Red Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Red Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
