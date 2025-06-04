import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  # Ganti ke 1 jika pakai USB camera eksternal

    if not cap.isOpened():
        print("Kamera gagal dibuka")
        return

    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA tersedia: {use_cuda}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))

        if use_cuda:
            # Upload frame ke GPU dan konversi ke HSV
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            hsv_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)

            # Unduh ke CPU untuk thresholding
            hsv_cpu = hsv_gpu.download()
        else:
            # Konversi di CPU
            hsv_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Thresholding warna merah di CPU
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_cpu, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_cpu, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Temukan kontur objek merah
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter noise kecil
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Red Object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Tampilkan hasil
        cv2.imshow("Red Object Tracking", frame)
        # cv2.imshow("Mask", red_mask)  # Aktifkan jika ingin lihat mask juga

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
