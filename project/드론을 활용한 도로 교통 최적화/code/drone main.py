from djitellopy import Tello
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import serial
import time
import threading
import queue

# 시리얼 통신 설정
ser = serial.Serial('/dev/ttyACM0', 9600)  # Linux/Mac 포트
# ser = serial.Serial('COM3', 9600)  # Windows 포트

# Tello 드론 초기화
tello = Tello()
tello.connect()
tello.streamon()

# YOLO 모델 로드
model = YOLO("/home/piai/PycharmProjects/Tello/pythonProject2/runs/detect/train/weights/best.pt")

# 카운팅할 라인 포인트와 클래스 정의
line_points = [
    [(20, 400), (1080, 400)],
    [(20, 50), (1080, 50)],
    [(80, 10), (80, 500)],
    [(550, 10), (550, 500)]
]

classes_to_count = [0]  # 카운팅할 클래스 인덱스 (예: 차량)

# ObjectCounter 초기화
counters = []
for reg_pts in line_points:
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False,
                     reg_pts=reg_pts,
                     classes_names=model.names,
                     draw_tracks=True,
                     line_thickness=2)
    counters.append(counter)

# 비디오 스트림 처리를 위한 스레드
frame_queue = queue.Queue(maxsize=4)  # 프레임 버퍼 큐

# 박스 필터링 함수, 차량에 해당하는 객체만 필터링
def filter_boxes(boxes, names):
    filtered_boxes = []
    for box_data in boxes.data:
        x1, y1, x2, y2, conf, cls_idx = box_data
        if cls_idx.item() == car_class_index:
            cls_name = names[int(cls_idx.item())]
            filtered_boxes.append((cls_name, conf.item(), [x1, y1, x2, y2]))
    return filtered_boxes

def video_stream():
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (600, 600))
        if frame_queue.full():  # 프레임 큐가 가득 차면 오래된 프레임 삭제
            frame_queue.get_nowait()
        frame_queue.put(frame)

thread = threading.Thread(target=video_stream)
thread.daemon = True
thread.start()

def control(key):
    if key == ord('q'):
        return False
    elif key == ord('t'):
        tello.takeoff()
    elif key == ord('f'):
        tello.move_forward(30)
    elif key == ord('b'):
        tello.move_back(30)
    elif key == ord('r'):
        tello.move_right(30)
    elif key == ord('l'):
        tello.move_left(30)
    elif key == ord('c'):
        tello.rotate_clockwise(30)
    elif key == ord('a'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('u'):
        tello.move_up(30)
    elif key == ord('d'):
        tello.move_down(30)
    return True

# 메인 루프
try:
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)

            for counter in counters:
                frame = counter.start_counting(frame, tracks)

            total_counts = [counter.in_counts + counter.out_counts for counter in counters]

            for i, count in enumerate(total_counts):
                print(f"Zone {i + 1} Count: {count}")
                ser.write((str(i+1) if count > 0 else '0').encode())

            cv2.imshow("Tello Object Counting", frame)

            # Zone 1에 차량이 있는지 확인 후 시리얼 통신으로 아두이노에 신호 보내기
            # 키보드 입력 받기
            key = cv2.waitKey(1) & 0xFF
            if not control(key):
                break

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
    ser.close()
    thread.join()
