import cv2
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=320,
    capture_height=240,
    display_width=320,
    display_height=240,
    framerate=60,
    flip_method=2,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def main():
    # Aktifkan kamera
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Gagal membuka kamera.")
        return

    print("Kamera terbuka. Mengukur FPS selama 100 frame...")

    frame_count = 0
    start_time = time.time()

    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        frame_count += 1

        # Uncomment untuk melihat preview (gunakan hanya untuk debugging!)
        # cv2.imshow("Preview", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    print(f"Captured {frame_count} frames in {elapsed_time:.2f} sec")
    print(f"Estimated FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
