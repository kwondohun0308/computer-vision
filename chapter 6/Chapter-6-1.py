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
