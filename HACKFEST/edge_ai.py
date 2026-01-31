import cv2
import requests
from datetime import datetime
from ultralytics import YOLO

# Load YOLO nano model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

API_URL = "http://127.0.0.1:5000/alert"

prev_gray = None

print("Edge AI started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame, conf=0.5)

    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "Person",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

    event = None

    # -------------------------------
    # ACTION LOGIC
    # -------------------------------

    # Crowd detection
    if person_count >= 4:
        event = "Crowd Formation Detected"

    # Motion detection
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_level = diff.mean()

        if motion_level > 20 and person_count >= 2:
            event = "Unusual Activity Detected"

    prev_gray = gray

    # -------------------------------
    # SEND ALERT
    # -------------------------------
    if event:
        data = {
            "camera_id": "CCTV-01",
            "event": event,
            "time": datetime.now().strftime("%H:%M:%S")
        }

        try:
            requests.post(API_URL, json=data, timeout=1)
        except:
            pass

        cv2.putText(frame, event,
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    cv2.imshow("Smart Campus Edge AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
