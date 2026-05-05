import cv2
import numpy as np
import threading
import queue
import time
import os
import collections
from classify import classify_img_path
from utils import remove_folder
from PytorchWildlife.models import classification as pw_classification
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "server"))
from database import init_db, insert_event

RTSP_URL       = "rtsp://192.168.1.201/profile2/media.smp"
FTP_DIR        = "ftp_uploads"
FPS            = 20
BUFFER_SECONDS = 30   # seconds of pre-event footage prepended to video
MOTION_TIMEOUT = 5   # seconds of no motion before event ends

classification_model = pw_classification.DFNE()

# Queues
frame_queue          = queue.Queue(maxsize=200)
motion_queue         = queue.Queue(maxsize=200)  # (frame, bboxes) from motion_detection -> event_manager
classification_queue = queue.Queue(maxsize=15)

stop_event = threading.Event()

# Pre-event rolling buffer
pre_event_buffer = collections.deque(maxlen=BUFFER_SECONDS * FPS)

# Event globals
current_event_dir      = None
current_classifications = []
current_event_frames   = []

# FTP trigger — set by ftp_watcher, cleared when event ends
motion_triggered = threading.Event()


# End of event handler
def end_event():
    global current_event_dir, current_classifications, current_event_frames

    print("\n--- Event Ended ---")

    if not current_classifications:
        print("No classifications recorded.")
        remove_folder(current_event_dir)
        print(f"Deleted empty event folder: {current_event_dir}")
        current_event_dir      = None
        current_classifications = []
        current_event_frames   = []
        motion_triggered.clear()
        return

    # Frequency + average confidence summary
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

    # Drain any remaining frames from motion_queue before saving video
    while not motion_queue.empty():
        frame, _ = motion_queue.get()
        current_event_frames.append(frame.copy())

    # Save video
    all_frames = list(pre_event_buffer) + current_event_frames
    if all_frames:
        h, w = all_frames[0].shape[:2]
        video_path = os.path.join(current_event_dir, "event.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
        for f in all_frames:
            if f.shape[:2] == (h, w):
                writer.write(f)
        writer.release()
        print(f"Video saved: {video_path}")
    else:
        print("No frames to write video.")

    print(f"Event folder: {current_event_dir}")
    print("-------------------\n")

    insert_event(
        folder_path=current_event_dir,
        timestamp=time.strftime("%Y-%m-%d %I:%M%p"),
        top_species=top3[0][0] if top3 else "Unknown",
        total_hits=len(current_classifications)
    )

    current_event_dir      = None
    current_classifications = []
    current_event_frames   = []
    motion_triggered.clear()


# Polls ftp_uploads for new jpg files — sets motion_triggered when found.
def ftp_watcher():
    os.makedirs(FTP_DIR, exist_ok=True)
    seen_files = set(os.listdir(FTP_DIR))
    print(f"FTP watcher started, watching: {os.path.abspath(FTP_DIR)}")

    while not stop_event.is_set():
        time.sleep(0.5)

        current_files = set(os.listdir(FTP_DIR))
        new_files = current_files - seen_files
        seen_files = current_files

        for filename in new_files:
            if not filename.lower().endswith(".jpg"):
                continue

            filepath = os.path.join(FTP_DIR, filename)

            # Wait for upload to finish
            prev_size = -1
            while True:
                try:
                    curr_size = os.path.getsize(filepath)
                except OSError:
                    break
                if curr_size == prev_size and curr_size > 0:
                    break
                prev_size = curr_size
                time.sleep(0.1)

            print(f"FTP trigger received: {filename}")

            if not motion_triggered.is_set():
                motion_triggered.set()
'''
            try:
                os.remove(filepath)
                seen_files.discard(filename)
            except OSError:
                pass
'''

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

# Runs MOG2 on every frame.
# Only forwards frames to motion_queue once FTP has triggered an event.
# Also maintains pre_event_buffer when no event is active.
def motion_detection():
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=1500,
        varThreshold=25,
        detectShadows=True
    )

    motion_counter   = 0
    motion_threshold = 8
    motion_active    = False
    last_motion_time = 0

    os.makedirs("motion_snaps", exist_ok=True)

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        # Always update pre-event buffer when no event is active
        if not motion_triggered.is_set():
            pre_event_buffer.append(frame.copy())

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        fgmask  = fgbg.apply(blurred)

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
        valid_bboxes    = []

        min_area = 1500
        ratio_x  = 0.3
        ratio_y  = 3.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / float(h)

            if ratio < ratio_x or ratio > ratio_y:
                continue
            if min(w, h) < 25:
                continue

            motion_detected = True
            valid_bboxes.append((x, y, w, h))

        # Keep the 2 largest bounding boxes
        valid_bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        valid_bboxes = valid_bboxes[:2]

        current_time = time.time()

        if motion_detected:
            motion_counter   = min(motion_counter + 1, motion_threshold + 1)
            last_motion_time = current_time
        else:
            motion_counter = max(0, motion_counter - 1)

        # Motion state — only relevant once FTP has triggered
        if motion_triggered.is_set():
            if not motion_active and motion_counter >= motion_threshold:
                motion_active = True
                print("Motion confirmed by MOG2")

            if motion_active and current_time - last_motion_time > MOTION_TIMEOUT:
                motion_active  = False
                motion_counter = 0
                print("Motion stopped — ending event")
                end_event()

        if not motion_triggered.is_set() or not motion_active or not motion_detected:
            continue

        if not motion_queue.full():
            motion_queue.put((frame.copy(), valid_bboxes))


# Creates folder, saves snapshots, accumulates frames for video.
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

        # Create event folder on first frame
        if current_event_dir is None:
            readable_time = time.strftime("%Y-%m-%d_%I-%M%p")
            folder_name   = f"event_{readable_time}"
            new_dir       = os.path.join("motion_snaps", folder_name)
            os.makedirs(new_dir, exist_ok=True)
            current_event_dir    = new_dir
            current_event_frames = []
            frame_counter        = 0
            print(f"Event folder created: {current_event_dir}")

        # Accumulate every frame for video
        current_event_frames.append(frame.copy())
        frame_counter += 1

        # Every FPS frames: save snapshot + send crops to classification
        if frame_counter % FPS == 0:
            timestamp = int(time.time() * 1000)

            # Save annotated full frame
            annotated = frame.copy()
            for (x, y, w, h) in bboxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            full_filename = os.path.join(current_event_dir, f"motion_{timestamp}_full.jpg")
            cv2.imwrite(full_filename, annotated)

            # Save zoom crop per bbox (largest 2) and send to classification
            for i, (x, y, w, h) in enumerate(bboxes):
                crop     = frame[y:y + h, x:x + w].copy()
                resized  = cv2.resize(crop, (224, 224))
                zoom_filename = os.path.join(current_event_dir, f"motion_{timestamp}_zoom_{i}.jpg")
                cv2.imwrite(zoom_filename, resized)

                if not classification_queue.full():
                    classification_queue.put(zoom_filename)


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


def main():
    init_db()
    threads = [
        threading.Thread(target=ftp_watcher,           name="FTPWatcher"),
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