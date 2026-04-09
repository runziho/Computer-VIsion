## 전체 코드
```python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# -------------------------------
# 1. 파일 경로 설정
# -------------------------------
video_path = "slow_traffic_small.mp4"
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"

# -------------------------------
# 2. YOLO 관련 설정값
# -------------------------------
conf_threshold = 0.5   # 객체라고 볼 최소 confidence
nms_threshold = 0.4    # NMS에 사용할 threshold
iou_threshold = 0.3    # tracker와 detection 매칭할 때 사용할 IoU 기준
max_missed = 10        # 몇 프레임 동안 안 잡히면 tracker 삭제할지

# COCO 데이터셋 클래스 이름
# yolov3 기본 모델이 보통 이 클래스 순서를 사용함
class_names = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]

# 이번 영상은 교통 영상이라 차량 관련 클래스만 추적
target_classes = ["car", "bus", "truck", "motorbike", "bicycle"]

# -------------------------------
# 3. IoU 계산 함수
#    두 박스가 얼마나 겹치는지 계산
# -------------------------------
def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area

# -------------------------------
# 4. Track 클래스
#    SORT에서 tracker 하나를 의미
#    여기서는 Kalman Filter를 이용해서
#    다음 위치를 예측함
# -------------------------------
class Track:
    next_id = 0

    def __init__(self, bbox):
        # tracker마다 고유 ID 부여
        self.id = Track.next_id
        Track.next_id += 1

        # 현재 bounding box 저장
        self.bbox = bbox

        # 몇 프레임 동안 detection과 매칭 안 됐는지 저장
        self.missed = 0

        # Kalman Filter 생성
        # 상태값 8개: x1, y1, x2, y2, vx1, vy1, vx2, vy2
        # 측정값 4개: x1, y1, x2, y2
        self.kf = cv2.KalmanFilter(8, 4)

        # 상태 전이 행렬
        # 위치 + 속도 모델
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # 측정 행렬
        # 실제로는 위치(x1, y1, x2, y2)만 관측 가능
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # 잡음 관련 행렬
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # 초기 상태 설정
        self.kf.statePost = np.array([
            [bbox[0]],
            [bbox[1]],
            [bbox[2]],
            [bbox[3]],
            [0],
            [0],
            [0],
            [0]
        ], dtype=np.float32)

    # 다음 위치 예측
    def predict(self):
        pred = self.kf.predict()

        x1 = int(pred[0][0])
        y1 = int(pred[1][0])
        x2 = int(pred[2][0])
        y2 = int(pred[3][0])

        self.bbox = [x1, y1, x2, y2]
        return self.bbox

    # detection 결과로 tracker 보정
    def update(self, bbox):
        measurement = np.array([
            [np.float32(bbox[0])],
            [np.float32(bbox[1])],
            [np.float32(bbox[2])],
            [np.float32(bbox[3])]
        ])

        self.kf.correct(measurement)
        self.bbox = bbox
        self.missed = 0

# -------------------------------
# 5. YOLO로 객체 검출하는 함수
#    현재 프레임에서 차량 객체를 찾아서
#    bounding box 리스트를 반환
# -------------------------------
def detect_objects(frame, net, output_layers):
    height, width = frame.shape[:2]

    # 이미지를 YOLO 입력 형태로 변환
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1/255.0,
        size=(608, 608),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # YOLO 출력 해석
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # confidence가 낮으면 무시
            if confidence < conf_threshold:
                continue

            label = class_names[class_id]

            # 차량 관련 클래스만 사용
            if label not in target_classes:
                continue

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # center 형식을 x1,y1,x2,y2 형태로 바꾸기 위해
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # NMS 적용해서 겹치는 박스 줄이기
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    result_boxes = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = max(0, x + w)
            y2 = max(0, y + h)

            result_boxes.append([x1, y1, x2, y2])

    return result_boxes

# -------------------------------
# 6. detection과 tracker를 매칭하는 함수
#    Hungarian 알고리즘 사용
# -------------------------------
def match_detections_and_tracks(detections, tracks):
    # tracker가 하나도 없으면
    # detection은 전부 새 tracker가 되어야 함
    if len(tracks) == 0:
        return [], list(range(len(detections))), []

    # IoU 행렬 만들기
    iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)

    for d in range(len(detections)):
        for t in range(len(tracks)):
            iou_matrix[d, t] = calc_iou(detections[d], tracks[t].bbox)

    # Hungarian 알고리즘은 비용 최소화이므로
    # IoU를 최대화하기 위해 -iou_matrix 사용
    row_idx, col_idx = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(tracks)))

    for r, c in zip(row_idx, col_idx):
        # IoU가 너무 낮으면 같은 객체로 안 봄
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            unmatched_detections.remove(r)
            unmatched_tracks.remove(c)

    return matches, unmatched_detections, unmatched_tracks

# -------------------------------
# 7. 메인 함수
# -------------------------------
def main():
    # YOLO 모델 불러오기
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    output_layers = net.getUnconnectedOutLayersNames()

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return

    tracks = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 1) 현재 프레임에서 객체 검출
        detections = detect_objects(frame, net, output_layers)

        # 2) 기존 tracker들 위치 예측
        for trk in tracks:
            trk.predict()
            trk.missed += 1

        # 3) detection과 tracker 매칭
        matches, unmatched_detections, unmatched_tracks = match_detections_and_tracks(detections, tracks)

        # 4) 매칭된 tracker는 detection으로 update
        for det_idx, trk_idx in matches:
            tracks[trk_idx].update(detections[det_idx])

        # 5) 매칭 안 된 detection은 새 tracker 생성
        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx])
            tracks.append(new_track)

        # 6) 너무 오래 못 찾은 tracker는 제거
        new_tracks = []
        for trk in tracks:
            if trk.missed <= max_missed:
                new_tracks.append(trk)
        tracks = new_tracks

        # 7) 결과 화면에 출력
        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox

            # 화면 밖으로 나가는 좌표 조금 보정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(0, x2)
            y2 = max(0, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                "ID {}".format(trk.id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 결과 영상 보여주기
        cv2.imshow("YOLOv3 + SORT Tracking", frame)

        # ESC 누르면 종료
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# 8. 실행
# -------------------------------
if __name__ == "__main__":
    main()
```
