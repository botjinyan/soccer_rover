from flask import Flask, Response
import cv2
from ultralytics import YOLO
import serial
import time

# ==== SERIAL ====
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # allow Arduino reset

# ==== YOLO ====
model = YOLO("best_ncnn_model")

# ==== CAMERA ====
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

app = Flask(__name__)

CENTER_X = 320

def send_command(mode, x, y):
    msg = f"{mode} {int(x)} {int(y)}\n"
    ser.write(msg.encode())
    print("SEND:", msg.strip())

BALL_CLASS = 2
LEFT_POST = 1
RIGHT_POST = 3
TOP_POST = 4
GOAL_AREA = 0

BALL_CLOSE_Y = 300   # ball near bottom of image = close to robot

def process_detections(results, frame):

    ball = None
    left = None
    right = None

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        xyxy = boxes.xyxy.cpu().numpy()
        cls  = boxes.cls.cpu().numpy()

        for box, c in zip(xyxy, cls):

            x1,y1,x2,y2 = box
            cx = (x1+x2)/2
            cy = (y1+y2)/2

            if int(c) == BALL_CLASS:
                ball = (cx,cy)

            elif int(c) == LEFT_POST:
                left = (cx,cy)

            elif int(c) == RIGHT_POST:
                right = (cx,cy)

    # ===== DECISION TREE =====

    # 1) NO BALL → SEARCH
    if ball is None:
        send_command(0,0,0)
        return

    # 2) BALL FAR → CHASE BALL
    if ball[1] < BALL_CLOSE_Y:
        send_command(1, ball[0], ball[1])
        return

    # 3) BALL CLOSE → AIM AT GOAL
    if left is not None and right is not None:
        goal_center_x = (left[0] + right[0]) / 2
        goal_center_y = (left[1] + right[1]) / 2

        send_command(2, goal_center_x, goal_center_y)
        return

    # fallback
    send_command(1, ball[0], ball[1])


def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, imgsz=320, conf=0.4, verbose=False)

        process_detections(results, frame)

        frame = results[0].plot()
        _, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')


@app.route('/')
def video():
    return Response(gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)
