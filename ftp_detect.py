import cv2
import threading
import queue
import time
import os
import collections
from classify import classify_img_path
from utils import remove_folder
from PytorchWildlife.models import classification as pw_classification

RTSP_URL      = "rtsp://192.168.1.201/profile2/media.smp"
FTP_DIR       = "ftp_uploads"           # Folder FileZilla Server deposits files into
FPS           = 15                      # Stream FPS — used for buffer size and classify cadence
BUFFER_SECONDS = 30                     # Seconds of pre-event footage to prepend to saved video
EVENT_SECONDS  = 30                     # Seconds of post-trigger footage to record before ending event

classification_model = pw_classification.DFNE()

# Queues
frame_queue        = queue.Queue(maxsize=200)  # raw frames from camera_capture -> frame_consumer
classification_queue = queue.Queue(maxsize=15) # image paths -> classification_worker

stop_event = threading.Event()

# Pre-event rolling buffer — always holds the last BUFFER_SECONDS * FPS raw frames
pre_event_buffer = collections.deque(maxlen=BUFFER_SECONDS * FPS)

# Event globals — written by frame_consumer / classification_worker, reset by end_event
current_event_dir     = None   # Path to the active event folder
current_classifications = []   # (label, confidence) tuples accumulated during event
current_event_frames  = []     # Every frame captured during the event (for video)

# Motion trigger flag — set by ftp_watcher, cleared by frame_consumer after event ends
motion_triggered = threading.Event()


def end_event():
    global current_event_dir, current_classifications, current_event_frames

    print("\n--- Event Ended ---")

    if not current_classifications:
        print("No classifications recorded.")
        remove_folder(current_event_dir)
        print(f"Deleted empty event folder: {current_event_dir}")
        current_event_dir     = None
        current_classifications = []
        current_event_frames  = []
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

    # Save video — pre-event buffer + event frames
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
        print(f"Video saved: {video_path} ({len(list(pre_event_buffer))} pre-event + {len(current_event_frames)} event frames)")
    else:
        print("No frames to write video.")

    print(f"Event folder: {current_event_dir}")
    print("-------------------\n")

    # Reset globals
    current_event_dir     = None
    current_classifications = []
    current_event_frames  = []


# FTP Watcher Thread
# Polls ftp_uploads folder for new jpg files — triggers a motion event
# when one appears. Cleans up the trigger file afterwards.
def ftp_watcher():
    os.makedirs(FTP_DIR, exist_ok=True)
    seen_files = set(os.listdir(FTP_DIR))
    print(f"FTP watcher started, watching: {os.path.abspath(FTP_DIR)}")

    while not stop_event.is_set():
        time.sleep(0.5)  # poll every 500ms

        current_files = set(os.listdir(FTP_DIR))
        new_files = current_files - seen_files
        seen_files = current_files

        for filename in new_files:
            if not filename.lower().endswith(".jpg"):
                continue

            filepath = os.path.join(FTP_DIR, filename)

            # Wait for file to finish uploading by checking size stabilises
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

            # Only trigger if no event is already active
            if not motion_triggered.is_set():
                motion_triggered.set()
            
            # Clean up trigger file
            #try:
                #os.remove(filepath)
                #seen_files.discard(filename)
            #except OSError:
            #    pass


# Camera Capture Thread
# Reads frames from RTSP stream and puts them on frame_queue.
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


# Frame Consumer Thread
# Always maintains the pre_event_buffer.
# When motion_triggered is set, starts an event — records for EVENT_SECONDS
# then calls end_event().
def frame_consumer():
    global current_event_dir, current_event_frames

    frame_counter = 0   # counts frames during active event (for classify cadence)
    event_active  = False
    event_frame_limit = EVENT_SECONDS * FPS  # total event frames before auto-end

    os.makedirs("motion_snaps", exist_ok=True)

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        # Always update pre-event buffer when no event is active
        if not event_active:
            pre_event_buffer.append(frame.copy())

        # Check for trigger
        if not event_active and motion_triggered.is_set():
            event_active = True
            frame_counter = 0
            current_event_frames = []

            readable_time = time.strftime("%Y-%m-%d_%I-%M%p")
            folder_name = f"event_{readable_time}"
            new_dir = os.path.join("motion_snaps", folder_name)
            os.makedirs(new_dir, exist_ok=True)
            current_event_dir = new_dir
            print(f"Event started, folder: {current_event_dir}")

        if not event_active:
            continue

        # Accumulate every frame for video
        current_event_frames.append(frame.copy())
        frame_counter += 1

        # Every FPS frames: save snapshot + send to classification
        if frame_counter % FPS == 0:
            timestamp = int(time.time() * 1000)

            # Save full frame snapshot
            snap_filename = os.path.join(current_event_dir, f"motion_{timestamp}_full.jpg")
            cv2.imwrite(snap_filename, frame)

            # Send full frame to classification
            if not classification_queue.full():
                classification_queue.put(snap_filename)

        # End event after EVENT_SECONDS of footage
        if frame_counter >= event_frame_limit:
            event_active = False
            motion_triggered.clear()
            print("Event duration reached — ending event")
            end_event()


# Classification Worker Thread
# Reads image paths from classification_queue, appends to current_classifications.
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
        threading.Thread(target=ftp_watcher,           name="FTPWatcher"),
        threading.Thread(target=camera_capture,        name="CameraCapture"),
        threading.Thread(target=frame_consumer,        name="FrameConsumer"),
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