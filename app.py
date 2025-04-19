import os, warnings, cv2, torch, pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]

people_history = {}

def log_history(id, first_seen, last_seen):
    out_path = "people_history.csv"
    df = pd.DataFrame([{"ID": id, "First Seen": first_seen, "Last Seen": last_seen}])
    df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)

cam_index = None
for i in range(5):
    cap_test = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap_test.isOpened():
        cam_index = i
        cap_test.release()
        break
    cap_test.release()

if cam_index is None:
    print("No camera found")
    exit()

cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"[ERROR] failed to grab frame (index={cam_index})")
        break

    results = model(frame)
    det = results.pandas().xyxy[0]
    people = det[det['name'] == 'person']
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for _, p in people.iterrows():
        x1, y1, x2, y2 = map(int, [p.xmin, p.ymin, p.xmax, p.ymax])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pid = f"{cx}-{cy}"

        if pid not in people_history:
            people_history[pid] = {'first_seen': now, 'last_seen': now}
        else:
            people_history[pid]['last_seen'] = now

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, pid, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("People Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for pid, times in people_history.items():
    log_history(pid, times['first_seen'], times['last_seen'])

cap.release()
cv2.destroyAllWindows()
