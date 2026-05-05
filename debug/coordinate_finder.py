import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")

rtsp_url = "rtsp://192.168.1.201/profile2/media.smp"
cap = cv2.VideoCapture(rtsp_url)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[75:, :]  # same crop as your system

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()