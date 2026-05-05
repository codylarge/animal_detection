import cv2

rtsp_url = "rtsp://192.168.1.201/profile2/media.smp"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to connect")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
