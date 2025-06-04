import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  # Ganti '0' jika pakai kamera eksternal

    if not cap.isOpened():
        print("Camera failed to open")
        return

    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA available: {use_cuda}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize agar proses lebih cepat
        frame = cv2.resize(frame, (640, 480))

        if use_cuda:
            # Upload frame ke GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Konversi ke HSV di GPU
            hsv_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)

            # Threshold warna merah di GPU
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.cuda.inRange(hsv_gpu, lower_red1, upper_red1)
            mask2 = cv2.cuda.inRange(hsv_gpu, lower_red2, upper_red2)
            red_mask = cv2.cuda.bitwise_or(mask1, mask2)

            # Convert back ke CPU untuk ditampilkan
            mask_cpu = red_mask.download()
            result = cv2.bitwise_and(frame, frame, mask=mask_cpu)

        else:
            # Fallback ke CPU
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)
            result = cv2.bitwise_and(frame, frame, mask=red_mask)

        # Tampilkan
        cv2.imshow("Red Object Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
