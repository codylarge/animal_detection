import cv2
from matplotlib.pyplot import gray
import numpy as np
import threading
import queue
import time
import os
import collections
from classify import classify_img_path
from utils import remove_folder
from PytorchWildlife.models import classification as pw_classification

RTSP_URL = "rtsp://192.168.1.201/profile2/media.smp"
FPS = 15                # Frames per second used to throttle classification, saving cadence, and buffer size
BUFFER_SECONDS = 30     # How many seconds of pre-event footage to prepend to the saved video

classification_model = pw_classification.DFNE()

# Queues
frame_queue = queue.Queue(maxsize=100)        # raw frames from camera -> motion detection
motion_queue = queue.Queue(maxsize=200)       # (frame, [bboxes]) tuples from motion detection -> event manager
classification_queue = queue.Queue(maxsize=15) # image paths from event manager -> classification worker

stop_event = threading.Event()

# Holds the last BUFFER_SECONDS * FPS frames
pre_event_buffer = collections.deque(maxlen=BUFFER_SECONDS * FPS)

# Event globals - written by event_manager and classification_worker, reset by end_motion_event
current_event_dir = None         # Path to the active event folder
current_classifications = []     # Growing list of classification label strings for the current event
current_event_frames = []        # Every frame from motion_queue during the event (for video)


# End-of-event handler — called by motion_detection when motion ends
def end_motion_event():
    global current_event_dir, current_classifications, current_event_frames

    print("\n--- Motion Event Ended ---")

    if not current_classifications:
        print("No classifications recorded for this event.")
        remove_folder(current_event_dir)
        print(f"Deleted event folder: {current_event_dir} due to no classifications")
        current_event_dir = None
        current_classifications = []
        current_event_frames = []
        return

    # Frequency summary
    freq = {}
    for label, confidence in current_classifications:
        if label not in freq:
            freq[label] = {"count": 0, "total_confidence": 0.0}
        freq[label]["count"] += 1
        freq[label]["total_confidence"] += confidence

    top3 = sorted(freq.items(), key=lambda x: x[1]["count"], reverse=True)[:3]

    print(f"Total classification hits: {len(current_classifications)}")
    print("Top species detected:")
    for label, data in top3:
        avg_conf = data["total_confidence"] / data["count"]
        print(f"  {label}: {data['count']} hit(s), avg confidence {avg_conf:.2f}")

    # Save results.txt
    results_path = os.path.join(current_event_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Total classification hits: {len(current_classifications)}\n")
        f.write("Top species detected:\n")
        for label, data in top3:
            avg_conf = data["total_confidence"] / data["count"]
            f.write(f"  {label}: {data['count']} hit(s), avg confidence {avg_conf:.2f}\n")

    # Save video 
    pre_frames = list(pre_event_buffer)
    all_frames = pre_frames + current_event_frames

    if all_frames:
        # NOTE: Frame dimensions come from cropped frames (fence crop applied in motion_detection).
        #       When crop is removed, dimensions will change automatically — no code change needed here.
        h, w = all_frames[0].shape[:2]
        video_path = os.path.join(current_event_dir, "event.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

        for f in all_frames:
            # Guard against any dimension mismatch at the crop boundary
            if f.shape[:2] == (h, w):
                writer.write(f)

        writer.release()
        print(f"Video saved: {video_path} ({len(pre_frames)} pre-event + {len(current_event_frames)} event frames)")
    else:
        print("No frames to write video.")

    print(f"Event folder: {current_event_dir}")
    print("--------------------------\n")

    # Reset globals
    current_event_dir = None
    current_classifications = []
    current_event_frames = []


# Camera Capture Thread
def camera_capture():
    while not stop_event.is_set():
        cap = cv2.VideoCapture(RTSP_URL)

        if not cap.isOpened():
            print("Camera connection failed. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print("Camera connected.")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Stream lost. Reconnecting...")
                break

            if not frame_queue.full():
                frame_queue.put(frame)

        cap.release()


# Motion Detection Thread
# Sends (frame, bboxes) tuples to motion_queue.
# bboxes is a list of up to 3 (x, y, w, h) tuples, largest first.
# Also maintains the pre_event_buffer with every processed frame.
def motion_detection():
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=1500,
        varThreshold=25,
        detectShadows=True
    )

    motion_counter = 0
    motion_threshold = 8
    motion_active = False

    last_motion_time = 0
    motion_timeout = 5

    os.makedirs("motion_snaps", exist_ok=True)

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        # Fence crop — removes top 75px to exclude static fence from motion detection.
        # TODO: Replace with a configurable exclusion zone system in the future.
        frame = frame[75:, :]

        # Add frame to buffer
        if not motion_active:
            pre_event_buffer.append(frame.copy())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        fgmask = fgbg.apply(blurred)

        # Detect floodlight / global change
        foreground_ratio = np.sum(fgmask > 0) / fgmask.size
        if foreground_ratio > 0.5:
            fgbg.apply(blurred, learningRate=1.0)
            print("Floodlight detected — resetting background")
            continue

        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        valid_bboxes = []

        min_area = 1000
        ratio_x = 0.3
        ratio_y = 3.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / float(h)

            if ratio < ratio_x or ratio > ratio_y:
                continue

            #if min(w, h) < 25:
            #    continue

            motion_detected = True
            valid_bboxes.append((x, y, w, h))

        # Keep the 3 largest bounding boxes
        valid_bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        valid_bboxes = valid_bboxes[:3]

        # Debug display
        #annotated = frame.copy()
        #for (x, y, w, h) in valid_bboxes:
        #    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow("Feed", annotated)
        #cv2.imshow("Mask", thresh)
        #cv2.waitKey(1)

        current_time = time.time()

        if motion_detected:
            motion_counter += 1
            last_motion_time = current_time
        else:
            motion_counter = max(0, motion_counter - 1)

        if not motion_active and motion_counter >= motion_threshold:
            motion_active = True
            print("Motion event started")

        if motion_active and current_time - last_motion_time > motion_timeout:
            motion_active = False
            motion_counter = 0
            print("Motion event ended")
            end_motion_event()

        if not motion_active or not motion_detected:
            continue

        if not motion_queue.full():
            motion_queue.put((frame.copy(), valid_bboxes))


# Event Manager Thread
# Owns: folder creation, snapshot saving, feeding classification_queue.
# Saves every frame for video, classifies / saves snapshots every FPS frames
def event_manager():
    global current_event_dir, current_event_frames

    frame_counter = 0

    while not stop_event.is_set():
        if motion_queue.empty():
            if current_event_dir is None and frame_counter != 0:
                frame_counter = 0
            time.sleep(0.01)
            continue

        frame, bboxes = motion_queue.get()

        # Create event folder on the very first frame of motion
        if current_event_dir is None:
            readable_time = time.strftime("%Y-%m-%d_%I-%M%p")
            folder_name = f"event_{readable_time}"
            new_dir = os.path.join("motion_snaps", folder_name)
            os.makedirs(new_dir, exist_ok=True)
            current_event_dir = new_dir
            current_event_frames = []
            frame_counter = 0
            print(f"Event folder created: {current_event_dir}")

        # Accumulate every frame for video
        current_event_frames.append(frame.copy())

        frame_counter += 1
        classify_interval = 1 # Classify interval in seconds

        # Every X frames: save snapshot + send to classification
        if frame_counter % (FPS * classify_interval) == 0:
            timestamp = int(time.time() * 1000)

            # Save annotated full frame
            annotated = frame.copy()
            for (x, y, w, h) in bboxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            full_filename = os.path.join(current_event_dir, f"motion_{timestamp}_full.jpg")
            cv2.imwrite(full_filename, annotated)

            # Save zoom crop per bbox, send each to classification
            for i, (x, y, w, h) in enumerate(bboxes):
                crop = frame[y:y + h, x:x + w].copy()
                resized = cv2.resize(crop, (224, 224))
                zoom_filename = os.path.join(current_event_dir, f"motion_{timestamp}_zoom_{i}.jpg")
                cv2.imwrite(zoom_filename, resized)

                if not classification_queue.full():
                    classification_queue.put(zoom_filename)


# Classification Worker Thread
# Reads from classification_queue, appends results to current_classifications
def classification_worker():
    global current_classifications

    while not stop_event.is_set():
        if classification_queue.empty():
            time.sleep(0.01)
            continue

        image_path = classification_queue.get()
        print(f"Classifying {image_path}...")

        result = classify_img_path(image_path, model=classification_model)

        if result is None:
            print("No species detected — skipping.")
            continue

        label, confidence = result
        print(f"Detected: {label} ({confidence:.2f})")
        current_classifications.append((label, confidence))


# Main
def main():
    threads = [
        threading.Thread(target=camera_capture,        name="CameraCapture"),
        threading.Thread(target=motion_detection,      name="MotionDetection"),
        threading.Thread(target=event_manager,         name="EventManager"),
        threading.Thread(target=classification_worker, name="ClassificationWorker"),
    ]

    for t in threads:
        t.daemon = True
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_event.set()

    for t in threads:
        t.join(timeout=5)


if __name__ == "__main__":
    main()