import cv2

# Inisialisasi kamera (0 untuk default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat diakses")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera")
        break

    # Tampilkan frame
    cv2.imshow("Kamera", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup
cap.release()
cv2.destroyAllWindows()
