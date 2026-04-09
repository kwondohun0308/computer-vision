import argparse
import os
import time

import cv2
import numpy as np


COCO_BRIEF = {0: "person", 2: "car"}
WINDOW_NAME = "YOLOv3 + SORT (Short)"
DISPLAY_SCALE = 2.0
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900


# ---------- YOLO detection ----------
def load_yolo(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    out_ids = net.getUnconnectedOutLayers()
    if len(out_ids.shape) == 1:
        out_names = [layer_names[i - 1] for i in out_ids]
    else:
        out_names = [layer_names[i[0] - 1] for i in out_ids]
    return net, out_names


def detect_objects(frame, net, out_names, conf_thres=0.5, nms_thres=0.4, size=416):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_names)

    boxes, scores, class_ids = [], [], []
    for output in outputs:
        for det in output:
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            score = float(np.max(cls_scores))
            if score < conf_thres:
                continue

            cx, cy = int(det[0] * w), int(det[1] * h)
            bw, bh = int(det[2] * w), int(det[3] * h)
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w - 1, x1 + bw), min(h - 1, y1 + bh)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)

    if not boxes:
        return np.empty((0, 6), dtype=np.float32)

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, nms_thres)
    if len(keep) == 0:
        return np.empty((0, 6), dtype=np.float32)

    detections = []
    for i in np.array(keep).flatten():
        x1, y1, x2, y2 = boxes[i]
        detections.append([x1, y1, x2, y2, scores[i], class_ids[i]])
    return np.array(detections, dtype=np.float32)


# ---------- SORT-style tracker ----------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-9 else inter / union


def iou_matrix(dets, trks):
    m = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            m[d, t] = iou(det[:4], trk)
    return m


class Track:
    next_id = 1

    def __init__(self, det):
        # state = [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.hstack([np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)])
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.id = Track.next_id
        Track.next_id += 1
        self.hits = 1
        self.miss = 0
        self.conf = float(det[4])
        self.class_id = int(det[5])

        self._set_state(det)

    def _xyxy_to_xywh(self, box):
        x1, y1, x2, y2 = box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        return np.array([cx, cy, w, h], dtype=np.float32)

    def _xywh_to_xyxy(self, x):
        cx, cy, w, h = map(float, x[:4])
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def _set_state(self, det):
        m = self._xyxy_to_xywh(det[:4]).reshape(4, 1)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4] = m

    def predict(self):
        x = self.kf.predict()
        self.miss += 1
        return self._xywh_to_xyxy(x.reshape(-1))

    def update(self, det):
        m = self._xyxy_to_xywh(det[:4]).reshape(4, 1)
        self.kf.correct(m)
        self.hits += 1
        self.miss = 0
        self.conf = float(det[4])
        self.class_id = int(det[5])

    def bbox(self):
        return self._xywh_to_xyxy(self.kf.statePost.reshape(-1))


class SortLite:
    def __init__(self, iou_thres=0.3, max_age=15, min_hits=3):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_idx = 0
        self.tracks = []

    def _associate(self, detections, predicted):
        if len(detections) == 0 or len(predicted) == 0:
            return [], list(range(len(detections))), list(range(len(predicted)))

        mat = iou_matrix(detections, predicted)
        pairs = []
        used_d, used_t = set(), set()

        # Greedy IoU matching: concise and enough for demo/presentation.
        while True:
            idx = np.unravel_index(np.argmax(mat), mat.shape)
            d, t = int(idx[0]), int(idx[1])
            best = float(mat[d, t])
            if best < self.iou_thres:
                break
            if d in used_d or t in used_t:
                mat[d, t] = -1.0
                continue
            pairs.append((d, t))
            used_d.add(d)
            used_t.add(t)
            mat[d, :] = -1.0
            mat[:, t] = -1.0

        u_dets = [i for i in range(len(detections)) if i not in used_d]
        u_trks = [i for i in range(len(predicted)) if i not in used_t]
        return pairs, u_dets, u_trks

    def update(self, detections):
        self.frame_idx += 1

        predicted = [trk.predict() for trk in self.tracks]
        pairs, u_dets, _ = self._associate(detections, predicted)

        for d, t in pairs:
            self.tracks[t].update(detections[d])

        for d in u_dets:
            self.tracks.append(Track(detections[d]))

        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]

        out = []
        for t in self.tracks:
            if t.miss == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits):
                x1, y1, x2, y2 = t.bbox()
                out.append([x1, y1, x2, y2, t.id, t.class_id, t.conf])
        return np.array(out, dtype=np.float32) if out else np.empty((0, 7), dtype=np.float32)


def color_for_id(track_id):
    rng = np.random.default_rng(seed=int(track_id) * 9991)
    c = rng.integers(0, 255, size=3, dtype=np.int32)
    return int(c[0]), int(c[1]), int(c[2])


def parse_args():
    p = argparse.ArgumentParser(description="Short YOLOv3 + SORT demo")
    p.add_argument("--video", default="slow_traffic_small.mp4")
    p.add_argument("--cfg", default="yolov3.cfg")
    p.add_argument("--weights", default="yolov3.weights")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.3)
    return p.parse_args()


def must_exist(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def main():
    args = parse_args()
    must_exist(args.video, "Video")
    must_exist(args.cfg, "YOLO cfg")
    must_exist(args.weights, "YOLO weights")

    net, out_names = load_yolo(args.cfg, args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    tracker = SortLite(iou_thres=args.iou, max_age=15, min_hits=3)
    prev = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detect_objects(frame, net, out_names, conf_thres=args.conf, nms_thres=args.nms)
        tracks = tracker.update(dets)
        for x1, y1, x2, y2, tid, cls_id, conf in tracks:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            tid, cls_id, conf = int(tid), int(cls_id), float(conf)
            color = color_for_id(tid)
            cls_name = COCO_BRIEF.get(cls_id, f"cls{cls_id}")
            label = f"ID {tid} {cls_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            fw, fh = frame.shape[1], frame.shape[0]
            tx = min(max(2, x1), max(2, fw - 260))
            ty = min(max(20, y1 - 8), fh - 6)

            # Clean label: black outline + colored text, no filled background.
            cv2.putText(
                frame,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        display = cv2.resize(
            frame,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import argparse
import os
import time

import cv2
import numpy as np

# 우리가 관심 있는 객체의 클래스 ID와 이름을 매핑합니다. (COCO 데이터셋 기준 0: 사람, 2: 자동차)
COCO_BRIEF = {0: "person", 2: "car"}

# 화면 출력용 설정값들입니다.
WINDOW_NAME = "YOLOv3 + SORT (Short)"
DISPLAY_SCALE = 2.0  # 화면 출력 크기 배율
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900


# ==========================================
# 1. YOLO 객체 탐지 (Detection) 파트
# ==========================================
def load_yolo(cfg_path, weights_path):
    """YOLO 네트워크를 메모리에 로드하고, 결과를 출력할 최종 레이어 이름을 찾습니다."""
    # 다크넷(Darknet) 구조의 YOLO 설정 파일과 가중치를 읽어옵니다.
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    
    # 네트워크에서 연결되지 않은 마지막 출력 레이어들의 인덱스를 가져와 이름을 추출합니다.
    out_ids = net.getUnconnectedOutLayers()
    if len(out_ids.shape) == 1:
        out_names = [layer_names[i - 1] for i in out_ids]
    else:
        out_names = [layer_names[i[0] - 1] for i in out_ids]
    return net, out_names


def detect_objects(frame, net, out_names, conf_thres=0.5, nms_thres=0.4, size=416):
    """현재 프레임에서 YOLO를 이용해 객체를 탐지하고, NMS로 중복을 제거합니다."""
    h, w = frame.shape[:2]
    
    # 이미지를 YOLO가 이해할 수 있는 4차원 텐서(Blob) 형태로 변환합니다.
    # 1/255.0으로 픽셀값을 정규화하고, 416x416 크기로 맞추며, BGR을 RGB로 바꿉니다(swapRB=True).
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 네트워크를 순전파(Forward) 시켜 탐지 결과들을 받아옵니다.
    outputs = net.forward(out_names)

    boxes, scores, class_ids = [], [], []
    # 출력된 결과물들을 하나씩 검사합니다.
    for output in outputs:
        for det in output:
            cls_scores = det[5:] # 0~4번은 박스 좌표 및 신뢰도, 5번부터가 클래스별 확률입니다.
            cls_id = int(np.argmax(cls_scores)) # 가장 확률이 높은 클래스를 찾습니다.
            score = float(np.max(cls_scores))   # 그 확률 값을 점수로 저장합니다.
            
            # 설정한 신뢰도(0.5)보다 낮으면 무시합니다.
            if score < conf_thres:
                continue

            # 박스의 중심점 좌표(cx, cy)와 너비(bw), 높이(bh)를 원본 이미지 비율에 맞게 복원합니다.
            cx, cy = int(det[0] * w), int(det[1] * h)
            bw, bh = int(det[2] * w), int(det[3] * h)
            
            # 중심점을 기준으로 박스의 좌상단(x1, y1)과 우하단(x2, y2) 좌표를 계산합니다.
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w - 1, x1 + bw), min(h - 1, y1 + bh)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)

    # 탐지된 객체가 없으면 빈 배열을 반환합니다.
    if not boxes:
        return np.empty((0, 6), dtype=np.float32)

    # NMS(비최대 억제)를 적용하기 위해 박스 형식을 [x, y, w, h]로 바꿉니다.
    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    
    # NMS를 실행하여 중복된 박스들을 지우고, 살아남은 박스의 인덱스(keep)만 가져옵니다.
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, nms_thres)
    if len(keep) == 0:
        return np.empty((0, 6), dtype=np.float32)

    # 살아남은 최종 박스들의 정보를 리스트로 정리하여 반환합니다.
    detections = []
    for i in np.array(keep).flatten():
        x1, y1, x2, y2 = boxes[i]
        detections.append([x1, y1, x2, y2, scores[i], class_ids[i]])
    return np.array(detections, dtype=np.float32)


# ==========================================
# 2. SORT 알고리즘 (객체 추적) 파트
# ==========================================
def iou(a, b):
    """두 바운딩 박스가 겹치는 비율(Intersection over Union)을 계산합니다."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    
    # 교집합(겹치는 부분)의 넓이
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    # 각 박스의 넓이를 구하고, 두 넓이를 더한 뒤 교집합을 빼서 합집합을 구합니다.
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    
    return 0.0 if union <= 1e-9 else inter / union


def iou_matrix(dets, trks):
    """모든 탐지된 박스(dets)와 추적 중인 박스(trks) 간의 IoU 점수 표(행렬)를 만듭니다."""
    m = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            m[d, t] = iou(det[:4], trk)
    return m


class Track:
    """개별 객체의 궤적(ID, 속도, 위치)을 관리하는 칼만 필터 클래스입니다."""
    next_id = 1 # 객체마다 부여할 고유 ID (1부터 시작)

    def __init__(self, det):
        # 상태 벡터: [중심x, 중심y, 너비, 높이, x속도, y속도, 너비변화속도, 높이변화속도] (총 8개)
        self.kf = cv2.KalmanFilter(8, 4)
        
        # 상태 전이 행렬: "다음 위치 = 현재 위치 + 속도"를 수학적으로 나타낸 '등속도 운동 모델'
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        # 측정 행렬 및 노이즈(오차) 설정
        self.kf.measurementMatrix = np.hstack([np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)])
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.id = Track.next_id
        Track.next_id += 1
        
        self.hits = 1       # 이 객체가 매칭에 성공한 횟수
        self.miss = 0       # 화면에서 사라진(놓친) 프레임 횟수
        self.conf = float(det[4])
        self.class_id = int(det[5])

        self._set_state(det)

    def _xyxy_to_xywh(self, box):
        """[좌상, 우하] 좌표를 칼만필터용 [중심x, 중심y, 너비, 높이]로 바꿉니다."""
        x1, y1, x2, y2 = box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        return np.array([cx, cy, w, h], dtype=np.float32)

    def _xywh_to_xyxy(self, x):
        """칼만필터용 좌표를 그리기 편한 [좌상, 우하] 좌표로 되돌립니다."""
        cx, cy, w, h = map(float, x[:4])
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def _set_state(self, det):
        """객체가 처음 발견되었을 때 칼만필터의 초기 상태를 세팅합니다."""
        m = self._xyxy_to_xywh(det[:4]).reshape(4, 1)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4] = m

    def predict(self):
        """칼만 필터의 물리학(등속도 모델)을 이용해 다음 프레임의 위치를 예측합니다."""
        x = self.kf.predict()
        self.miss += 1 # 일단 놓쳤다고 가정하고, 나중에 매칭되면 0으로 초기화합니다.
        return self._xywh_to_xyxy(x.reshape(-1))

    def update(self, det):
        """YOLO가 실제로 찾아낸 위치(det)를 입력받아 예측값의 오차를 수정(교정)합니다."""
        m = self._xyxy_to_xywh(det[:4]).reshape(4, 1)
        self.kf.correct(m)
        self.hits += 1     # 연속으로 잘 찾음
        self.miss = 0      # 놓친 횟수 초기화
        self.conf = float(det[4])
        self.class_id = int(det[5])

    def bbox(self):
        """현재 추적 중인 객체의 최적화된 박스 좌표를 반환합니다."""
        return self._xywh_to_xyxy(self.kf.statePost.reshape(-1))


class SortLite:
    """예측된 궤적과 새로 탐지된 객체를 짝지어주는 매니저(Tracker) 클래스입니다."""
    def __init__(self, iou_thres=0.3, max_age=15, min_hits=3):
        self.iou_thres = iou_thres # 최소한 이 정도는 겹쳐야 같은 객체로 인정함
        self.max_age = max_age     # 몇 프레임 동안 안 보이면 추적을 포기할지 (수명)
        self.min_hits = min_hits   # 처음 발견되고 최소 몇 번 연속 보여야 진짜로 믿을지
        self.frame_idx = 0
        self.tracks = []           # 현재 추적 중인 객체들의 리스트

    def _associate(self, detections, predicted):
        """YOLO의 새 탐지 결과(detections)와 칼만필터의 예측값(predicted)을 짝지어줍니다."""
        if len(detections) == 0 or len(predicted) == 0:
            return [], list(range(len(detections))), list(range(len(predicted)))

        # 두 그룹 간의 겹침(IoU) 행렬을 구합니다.
        mat = iou_matrix(detections, predicted)
        pairs = []
        used_d, used_t = set(), set()

        # 가장 많이 겹치는 짝부터 우선적으로(Greedy) 연결합니다.
        while True:
            idx = np.unravel_index(np.argmax(mat), mat.shape)
            d, t = int(idx[0]), int(idx[1])
            best = float(mat[d, t])
            
            # 가장 많이 겹치는 짝이 기준치(0.3)보다 낮으면 매칭을 중단합니다.
            if best < self.iou_thres:
                break
            # 이미 짝을 찾은 객체라면 패스합니다.
            if d in used_d or t in used_t:
                mat[d, t] = -1.0
                continue
            
            # 성공적으로 짝을 찾았습니다.
            pairs.append((d, t))
            used_d.add(d)
            used_t.add(t)
            # 짝을 찾은 행과 열은 지워버립니다.
            mat[d, :] = -1.0
            mat[:, t] = -1.0

        # 짝을 찾지 못한 새로운 탐지 객체(u_dets)와, 주인을 잃어버린 과거의 예측(u_trks)을 분류합니다.
        u_dets = [i for i in range(len(detections)) if i not in used_d]
        u_trks = [i for i in range(len(predicted)) if i not in used_t]
        return pairs, u_dets, u_trks

    def update(self, detections):
        """매 프레임마다 트래커를 업데이트하는 메인 함수입니다."""
        self.frame_idx += 1

        # 1. 기존에 추적 중이던 객체들이 지금 어디쯤 있을지 예측합니다.
        predicted = [trk.predict() for trk in self.tracks]
        
        # 2. 예측된 위치와 YOLO가 새로 찾은 위치를 매칭합니다.
        pairs, u_dets, _ = self._associate(detections, predicted)

        # 3. 매칭에 성공한 객체들은 YOLO의 실제 좌표로 궤도를 수정(Update)합니다.
        for d, t in pairs:
            self.tracks[t].update(detections[d])

        # 4. 새로 나타난 객체(짝이 없는 녀석들)는 새로운 추적 객체로 등록합니다.
        for d in u_dets:
            self.tracks.append(Track(detections[d]))

        # 5. 수명(max_age=15)이 다 된 객체, 즉 오랫동안 화면에 안 보이는 객체는 메모리에서 삭제합니다.
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]

        # 6. 화면에 출력할 최종 확정된 객체들의 정보를 모아 반환합니다.
        out = []
        for t in self.tracks:
            # 방금 발견되었고(miss==0), 연속으로 충분히 보인(hits>=min_hits) 확실한 객체만 반환합니다.
            if t.miss == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits):
                x1, y1, x2, y2 = t.bbox()
                out.append([x1, y1, x2, y2, t.id, t.class_id, t.conf])
        return np.array(out, dtype=np.float32) if out else np.empty((0, 7), dtype=np.float32)


# ==========================================
# 3. 유틸리티 및 메인 실행 파트
# ==========================================
def color_for_id(track_id):
    """객체 ID마다 고유한 박스 색상을 만들어주는 함수입니다."""
    rng = np.random.default_rng(seed=int(track_id) * 9991)
    c = rng.integers(0, 255, size=3, dtype=np.int32)
    return int(c[0]), int(c[1]), int(c[2])


def parse_args():
    """실행 시 터미널에서 옵션(가중치, 비디오 경로 등)을 받을 수 있게 해줍니다."""
    p = argparse.ArgumentParser(description="Short YOLOv3 + SORT demo")
    p.add_argument("--video", default="slow_traffic_small.mp4")
    p.add_argument("--cfg", default="yolov3.cfg")
    p.add_argument("--weights", default="yolov3.weights")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.3)
    return p.parse_args()


def must_exist(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def main():
    args = parse_args()
    must_exist(args.video, "Video")
    must_exist(args.cfg, "YOLO cfg")
    must_exist(args.weights, "YOLO weights")

    # 1. YOLO 모델을 메모리에 올립니다.
    net, out_names = load_yolo(args.cfg, args.weights)
    
    # 2. 분석할 동영상 파일을 엽니다.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    # 3. 추적기(SORT) 객체를 생성합니다.
    tracker = SortLite(iou_thres=args.iou, max_age=15, min_hits=3)
    prev = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    # 비디오의 프레임을 하나씩 읽으며 무한 반복합니다.
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # A. 현재 프레임에서 YOLO로 객체들을 찾습니다.
        dets = detect_objects(frame, net, out_names, conf_thres=args.conf, nms_thres=args.nms)
        
        # B. 찾은 객체들을 SORT 알고리즘에 넘겨주어 이전 객체와 연결(Tracking)합니다.
        tracks = tracker.update(dets)
        
        # C. 추적된 정보(ID, 클래스, 좌표 등)를 화면에 그립니다.
        for x1, y1, x2, y2, tid, cls_id, conf in tracks:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            tid, cls_id, conf = int(tid), int(cls_id), float(conf)
            
            color = color_for_id(tid) # ID에 맞는 고유 색상 가져오기
            cls_name = COCO_BRIEF.get(cls_id, f"cls{cls_id}")
            label = f"ID {tid} {cls_name} {conf:.2f}"

            # 바운딩 박스를 그립니다.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            fw, fh = frame.shape[1], frame.shape[0]
            tx = min(max(2, x1), max(2, fw - 260))
            ty = min(max(20, y1 - 8), fh - 6)

            # 글씨가 잘 보이도록 까만색 테두리를 먼저 그리고, 그 위에 색상 글씨를 덮어씁니다. (Clean label)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 현재 프레임의 연산 속도(FPS)를 계산하여 좌측 상단에 표시합니다.
        now = time.time()
        fps = 1.0 / max(1e-6, now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 화면 크기를 키워서 보여줍니다.
        display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(WINDOW_NAME, display)
        
        # 'q'나 'ESC' 키를 누르면 영상을 종료합니다.
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    # 사용한 자원을 안전하게 반납합니다.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
