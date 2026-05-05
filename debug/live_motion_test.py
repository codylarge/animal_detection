import cv2
import numpy as np

rtsp_url = "rtsp://192.168.1.201/profile2/media.smp"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to connect")
    exit()

# Adjustable parameters
history = 500
varThreshold = 25
min_area = 1500

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=history,
    varThreshold=varThreshold,
    detectShadows=False
)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    fgmask = fgbg.apply(blurred)
    _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Live Motion Test", frame)
    cv2.imshow("Foreground Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()