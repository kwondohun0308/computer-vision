# chapter 6
## 과제1 설명 및 요구사항 (SORT 알고리즘을 활용한 다중 객체 추적기 구현)
 - 이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현합니다. 이를 통해 객체 추적의 기본개념과 SORT 알고리즘의 적용방법을 학습할 수 있습니다.
 - 객체 검출기 구현: YOLOv3와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출합니다.​
 - mathworks.comSORT 추적기 초기화: 검출된 객체의 경계상자를 입력으로 받아 SORT 추적기를 초기화합니다.​
 - 객체추적: 각 프레임마다 검출된 객체와 기존추적객체를 연관시켜 추적을 유지합니다.
 - 결과 시각화: 추적된 각 객체에 고유 ID를부여하고, 해당 ID와 경계상자를 비디오 프레임에 표시하여 실시간으로 출력합니다.

과제 한줄 요약 - YOLOv3를 통해 영상 속 객체를 실시간으로 검출하고, SORT 알고리즘을 결합해 각 객체에 고유 ID를 부여하며 연속적인 이동 궤적을 따라가는 다중 객체 추적(MOT) 시스템 구현 과제

<details>
	<summary>과제 1 전체 코드</summary>
	
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


</details>


![1번 결과](https://github.com/user-attachments/assets/586c6adc-0e98-4b39-a502-73f126253134)



## 과제 1 주요 코드 설명
1. YOLO 객체 탐지 (Detection): 매 프레임마다 대상 찾기

	```python
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
	```
	가장 먼저 영상의 매 프레임마다 YOLO 모델을 가동하여 사람과 자동차의 위치(바운딩 박스)를 찾아냅니다.
	이때 하나의 객체에 여러 개의 박스가 중복으로 그려지는 현상을 막기 위해, NMS(Non-Maximum Suppression) 알고리즘을 거쳐 가장 확률이 높은 깔끔한 박스 하나만 남겨둡니다.
	NMS(Non-Maximum Suppression) 알고리즘은 가장 확실한 정답 하나만 남기고, 주변에 얼쩡거리는 비슷한 오답들을 깔끔하게 지워버리는 알고리즘 입니다.
	

2. 칼만 필터 (Kalman Filter): 물리학 기반의 위치 예측
   ```python
				self.kf = cv2.KalmanFilter(8, 4)
		        self.kf.transitionMatrix = np.array(
		            [
		                [1, 0, 0, 0, 1, 0, 0, 0], #칼만 필터의 상태 전이 행렬
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
   ```
   이 코드의 핵심인 객체 추적(Tracking)을 위해 칼만 필터(Kalman Filter)를 도입했습니다.
   
   칼만 필터는 객체의 현재 위치뿐만 아니라 이동 속도와 방향을 함께 기억합니다.
   
   이를 통해 다음 프레임에서 이 자동차가 어디쯤 이동해 있을지 물리적으로 미리 예측(predict)하고, 실제 YOLO가 찾은 위치로 오차를 보정하며 부드럽고 정확한 추적 궤적을 만들어냅니다.

3. SORT 매칭 (IoU Association): 과거와 현재 짝지어주기
   ```python
		def iou_matrix(dets, trks):
		    m = np.zeros((len(dets), len(trks)), dtype=np.float32)
		    for d, det in enumerate(dets):
		        for t, trk in enumerate(trks):
		            m[d, t] = iou(det[:4], trk)
		    return m
		(중략...)
   		mat = iou_matrix(detections, predicted)
   ```
   프레임이 넘어가면 1번 객체와 2번 객체가 누구인지 컴퓨터는 잊어버리므로 이를 해결하기 위해, 방금 전 칼만 필터가 '예측한 위치'와 현재 YOLO가 '실제로 찾은 위치'가 얼마나 겹치는지 IoU(교집합/합집합 비율)를 계산합니다.

   겹치는 면적이 가장 넓은 박스들끼리 동일한 객체로 판명하고 기존의 ID(예: ID 1번)를 그대로 유지해 줍니다.

4. 생애 주기 관리 (Life Cycle): 메모리 최적화
   ```python
		self.tracks = [t for t in self.tracks if t.miss <= self.max_age]
   ```
   실시간 영상에서는 객체가 카메라 밖으로 나가거나 장애물에 가려지는 일이 빈번합니다.

   YOLO가 연속으로 15프레임(max_age=15) 이상 찾지 못한 객체는 화면에서 사라졌다고 판단하여 추적 리스트에서 영구 삭제함으로써 시스템의 메모리와 연산 속도를 가볍게 유지합니다.
   
## 과제2 설명 및 요구사항 (Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화)
 - Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고, 이를 실시간 영상에 시각화하는 프로그램을 구현합니다.​
 - Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화합니다.
 - OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처합니다.​
 - 검출된 얼굴랜드 마크를 실시간 영상에 점으로 표시합니다.​
 - ESC 키를 누르면 프로그램이 종료되도록 설정합니다.

과제 한줄 요약 - Google의 MediaPipe 라이브러리를 활용하여 단일 이미지에서 얼굴의 468개 특징점(Landmarks)을 추출하고, 이를 원본 이미지 위에 맵핑하여 시각화하는 얼굴 인식 파이프라인 구현 과제

<details>
	<summary>과제 2 전체 코드</summary>
	
		import cv2
		import mediapipe as mp
		import os
		import urllib.request
		import ctypes
		
		# 신버전 API(Tasks)를 사용할 때 필요한 인공지능 모델 파일의 다운로드 주소입니다.
		MODEL_URL = (
		    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
		    "face_landmarker/float16/latest/face_landmarker.task"
		)
		
		# ==========================================
		# 1. 유틸리티 (도구) 함수 파트
		# ==========================================
		def ensure_face_landmarker_model(model_path):
		    """신버전 API 작동에 필요한 .task 모델 파일이 없으면 자동으로 다운로드합니다."""
		    # 파일이 이미 존재하면 다운로드하지 않고 넘어갑니다.
		    if os.path.exists(model_path):
		        return
		
		    # 폴더가 없으면 만들고, 구글 서버에서 모델을 다운로드하여 저장합니다.
		    os.makedirs(os.path.dirname(model_path), exist_ok=True)
		    print(f"Downloading model to: {model_path}")
		    urllib.request.urlretrieve(MODEL_URL, model_path)
		
		
		def draw_landmarks(frame, landmarks):
		    """찾아낸 468개의 얼굴 특징점 좌표에 초록색 점을 찍어주는 함수입니다."""
		    h, w = frame.shape[:2]
		    
		    # 이미지 해상도에 맞춰 점의 크기(반지름)를 1~3픽셀 사이로 유동적으로 조절합니다.
		    radius = max(1, min(3, int(round(min(h, w) / 320))))
		    
		    # MediaPipe는 기본적으로 얼굴 형태 468개 + 눈동자(Iris) 10개 등 여러 점을 줍니다.
		    # 여기서는 얼굴의 기본 골격인 468개만 잘라서 그립니다.
		    for lm in landmarks[:468]:
		        # MediaPipe가 주는 좌표(lm.x, lm.y)는 0.0 ~ 1.0 사이의 비율(정규화) 값입니다.
		        # 따라서 실제 이미지의 가로(w), 세로(h) 길이를 곱해 '실제 픽셀 좌표'로 변환합니다.
		        x = int(lm.x * w)
		        y = int(lm.y * h)
		        
		        # 계산된 좌표가 이미지 화면 안에 정상적으로 들어오는지 확인 후 점을 찍습니다.
		        if 0 <= x < w and 0 <= y < h:
		            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
		
		
		def resize_to_fit_screen(image, margin=120):
		    """사진이 모니터 화면보다 너무 클 경우, 화면에 맞게 자동으로 축소해주는 UX 개선 함수입니다."""
		    try:
		        # 윈도우 OS의 기능을 빌려와 현재 모니터의 해상도(너비, 높이)를 파악합니다.
		        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
		        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
		    except Exception:
		        # 파악에 실패하면 기본 해상도(1280x720)로 설정합니다.
		        screen_w, screen_h = 1280, 720
		
		    # 여백(margin)을 뺀 최대 허용 크기를 구합니다.
		    max_w = max(320, screen_w - margin)
		    max_h = max(240, screen_h - margin)
		
		    h, w = image.shape[:2]
		    # 가로, 세로 중 더 많이 줄여야 하는 쪽의 축소 비율(scale)을 찾습니다.
		    scale = min(max_w / w, max_h / h, 1.0)
		
		    # 1.0보다 작다는 것은 사진이 모니터보다 크다는 뜻이므로 축소합니다.
		    if scale < 1.0:
		        new_w = int(w * scale)
		        new_h = int(h * scale)
		        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		
		    return image
		
		
		# ==========================================
		# 2. MediaPipe 얼굴 인식 코어 파트 (두 가지 방식 호환)
		# ==========================================
		def run_with_solutions_face_mesh(image):
		    """[구버전 API] 간편하게 사용할 수 있는 기존의 Solutions API 방식입니다."""
		    mp_face_mesh = mp.solutions.face_mesh
		
		    # 얼굴 인식 모델을 세팅합니다.
		    # static_image_mode=True: 동영상이 아닌 '단일 이미지'용으로 정밀하게 분석합니다.
		    with mp_face_mesh.FaceMesh(
		        static_image_mode=True,
		        max_num_faces=1,             # 최대 찾을 얼굴 개수
		        refine_landmarks=True,       # 눈동자나 입술 주변을 더 정밀하게 찾을지 여부
		        min_detection_confidence=0.5,# 50% 이상 확신이 들 때만 얼굴로 판정
		        min_tracking_confidence=0.5,
		    ) as face_mesh:
		        
		        # OpenCV의 BGR 색상을 MediaPipe가 좋아하는 RGB 색상으로 바꿔줍니다.
		        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		        
		        # 모델에 이미지를 넣어 얼굴 특징점을 추출(process)합니다.
		        results = face_mesh.process(rgb)
		
		        output = image.copy()
		        # 얼굴을 하나라도 찾았다면?
		        if results.multi_face_landmarks:
		            # 찾은 모든 얼굴(여기서는 max_num_faces=1이므로 1개)에 대해 점을 그립니다.
		            for face_landmarks in results.multi_face_landmarks:
		                draw_landmarks(output, face_landmarks.landmark)
		
		        return output
		
		
		def run_with_tasks_face_landmarker(image):
		    """[신버전 API] 더 강력하고 최적화된 최신의 Tasks API 방식입니다."""
		    # 모델(.task) 파일의 위치를 지정하고, 없으면 다운로드합니다.
		    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
		    ensure_face_landmarker_model(model_path)
		
		    BaseOptions = mp.tasks.BaseOptions
		    FaceLandmarker = mp.tasks.vision.FaceLandmarker
		    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
		    VisionRunningMode = mp.tasks.vision.RunningMode
		
		    # 모델 옵션을 설정합니다. (단일 이미지 모드, 최대 얼굴 1개)
		    options = FaceLandmarkerOptions(
		        base_options=BaseOptions(model_asset_path=model_path),
		        running_mode=VisionRunningMode.IMAGE,
		        num_faces=1,
		    )
		
		    # 설정된 옵션으로 얼굴 인식 모델을 생성합니다.
		    with FaceLandmarker.create_from_options(options) as landmarker:
		        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		        # 이미지를 Tasks API 전용 포맷(mp.Image)으로 변환합니다.
		        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		        
		        # 예측(detect)을 수행합니다.
		        result = landmarker.detect(mp_image)
		
		        output = image.copy()
		        if result.face_landmarks:
		            for face_landmarks in result.face_landmarks:
		                draw_landmarks(output, face_landmarks)
		
		        return output
		
		
		# ==========================================
		# 3. 메인 실행 파트
		# ==========================================
		def main():
		    # 분석할 얼굴 사진의 경로를 가져옵니다. (코드와 같은 폴더에 있어야 함)
		    image_path = os.path.join(
		        os.path.dirname(__file__),
		        "asian-man-isolated-expressing-emotions.jpg",
		    )
		
		    # 파일이 없는 경우 에러를 띄웁니다.
		    if not os.path.exists(image_path):
		        raise RuntimeError("Image file not found: " + image_path)
		
		    # OpenCV로 이미지를 읽어옵니다.
		    image = cv2.imread(image_path)
		    if image is None:
		        raise RuntimeError("Failed to load image: " + image_path)
		
		    # 설치된 MediaPipe 버전에 따라 알맞은 API를 자동으로 선택하여 실행합니다.
		    # (과거 코드가 최신 라이브러리에서 작동하지 않는 것을 방지하기 위한 아주 안전한 설계입니다)
		    try:
		        # 만약 'solutions.face_mesh'라는 구버전 모듈이 존재하면 그것을 사용하고,
		        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
		            output = run_with_solutions_face_mesh(image)
		        # 없으면 최신 'Tasks' API를 사용합니다.
		        else:
		            output = run_with_tasks_face_landmarker(image)
		    except Exception as exc:
		        raise RuntimeError(
		            "Face landmark initialization failed. "
		            "If using modern MediaPipe builds, this script uses tasks API fallback. "
		            f"Original error: {exc}"
		        )
		
		    # 모니터에 출력할 창을 띄우는 무한 루프입니다.
		    while True:
		        # 이미지가 너무 크면 줄여서 화면에 띄웁니다.
		        display = resize_to_fit_screen(output)
		        cv2.imshow("MediaPipe FaceMesh (468 Landmarks)", display)
		        
		        # 키보드 입력 대기 (30ms) -> 'ESC' 키(아스키코드 27)를 누르면 종료합니다.
		        key = cv2.waitKey(30) & 0xFF
		        if key == 27:
		            break
		
		    # 창을 안전하게 닫습니다.
		    cv2.destroyAllWindows()
		
		
		if __name__ == "__main__":
		    main()


</details>

![2번 결과](https://github.com/user-attachments/assets/77c7e203-d061-4c65-8daf-458226b820a6)



## 과제 2 주요 코드 설명

1. API 호환성 안전망: 과거와 현재 버전 모두 지원
	```python
   		try:
        # 1. 만약 사용자의 파이썬에 '구버전(solutions)'이 설치되어 있다면
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            output = run_with_solutions_face_mesh(image)
            
        # 2. 구버전이 없고 최신 버전만 있다면 '신버전(tasks)'으로 실행
        else:
            output = run_with_tasks_face_landmarker(image)
            
    except Exception as exc:
        raise RuntimeError(f"Face landmark 초기화 실패: {exc}")
 	```
	구글 MediaPipe가 최근 버전 업데이트를 하면서 코드가 호환되지 않는 이슈가 많습니다.
	
	그래서 어떤 실행 환경에서도 프로그램이 멈추지 않고 완벽하게 돌아가도록, 구버전과 신버전 API를 모두 구현해두고 상황에 맞게 자동 분기되도록 설계했습니다

2. MediaPipe 코어 엔진: 468개 특징점 추출
	```python
 		# 1. 모델 옵션 설정: 단일 이미지(IMAGE)에서 뚜렷한 얼굴 1개(num_faces=1)를 찾음
    	options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # 2. OpenCV의 BGR 이미지를 AI가 인식할 수 있는 전용 포맷(mp.Image)으로 변환
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # 3. 468개의 3D 좌표 데이터를 추출
        result = landmarker.detect(mp_image)
 	```
	실제 얼굴 인식을 수행하는 코어 엔진입니다. 미리 학습된 AI 모델을 불러온 뒤, 이미지를 전용 포맷으로 변환해 detect 함수에 넣습니다.
	
	그러면 단 1줄의 코드로 눈썹의 굴곡, 입술의 두께, 턱선의 각도까지 담아낸 468개의 3D 랜드마크 데이터가 추출됩니다.

3. 좌표 정규화 및 시각화: 비율을 실제 픽셀로 환산
	```python
			def draw_landmarks(frame, landmarks):
		    # 이미지의 실제 세로(h), 가로(w) 길이를 가져옴
		    h, w = frame.shape[:2]
		    
		    # 468개의 점 데이터를 하나씩 꺼내어 반복
		    for lm in landmarks[:468]:
		        # [핵심] 0.0~1.0 사이의 비율 좌표(lm.x, lm.y)에 실제 해상도를 곱해 픽셀 위치로 변환
		        x = int(lm.x * w)
		        y = int(lm.y * h)
		        
		        # 계산된 픽셀 위치에 초록색 점(Circle) 그리기
		        if 0 <= x < w and 0 <= y < h:
		            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
 	```
	추출된 데이터를 사진 위에 그리는 부분입니다. MediaPipe는 얼굴 좌표를 300픽셀 같은 절대값이 아니라 가로의 50%, 세로의 20% (0.5, 0.2) 같은 정규화된 비율 값으로 알려줍니다.
	
	이 비율에 실제 사진의 가로/세로 해상도를 곱하여 픽셀 좌표로 환산했습니다.

