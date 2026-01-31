import cv2
import requests
import time
from datetime import datetime
from ultralytics import YOLO

# ----------------------------
# Load YOLO model (lightweight)
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# CCTV / Webcam input
# ----------------------------
cap = cv2.VideoCapture(0)

# ----------------------------
# API endpoint
# ----------------------------
API_URL = "http://127.0.0.1:5000/alert"

# ----------------------------
# Variables
# ----------------------------
prev_gray = None
last_alert_time = 0
ALERT_COOLDOWN = 5   # seconds

print("Edge AI started successfully...")

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for motion analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run YOLO detection
    results = model(frame, conf=0.5)

    person_count = 0

    # ----------------------------
    # Person detection
    # ----------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # Class 0 = person
            if cls == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    event = None

    # ----------------------------
    # Crowd detection
    # ----------------------------
    if person_count >= 4:
        event = "Crowd Formation Detected"

    # ----------------------------
    # Motion-based unusual activity
    # ----------------------------
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_level = diff.mean()

        # Trigger ONLY when multiple people + high motion
        if motion_level > 25 and person_count >= 2:
            event = "Unusual Activity Detected"

    prev_gray = gray

    # ----------------------------
    # Alert sending with cooldown
    # ----------------------------
    current_time = time.time()

    if event and (current_time - last_alert_time > ALERT_COOLDOWN):
        last_alert_time = current_time

        data = {
            "camera_id": "CCTV-01",
            "event": event,
            "time": datetime.now().strftime("%H:%M:%S")
        }

        try:
            requests.post(API_URL, json=data, timeout=1)
        except:
            pass

        # Show alert text on screen
        cv2.putText(
            frame,
            event,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ----------------------------
    # Display output
    # ----------------------------
    cv2.imshow("Smart Campus Edge AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
