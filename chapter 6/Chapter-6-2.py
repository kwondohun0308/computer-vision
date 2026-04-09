import cv2
import mediapipe as mp
import os
import urllib.request
import ctypes


MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
	"face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_face_landmarker_model(model_path):
	if os.path.exists(model_path):
		return

	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	print(f"Downloading model to: {model_path}")
	urllib.request.urlretrieve(MODEL_URL, model_path)


def draw_landmarks(frame, landmarks):
	h, w = frame.shape[:2]
	radius = max(1, min(3, int(round(min(h, w) / 320))))
	for lm in landmarks[:468]:
		x = int(lm.x * w)
		y = int(lm.y * h)
		if 0 <= x < w and 0 <= y < h:
			cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)


def resize_to_fit_screen(image, margin=120):
	try:
		screen_w = ctypes.windll.user32.GetSystemMetrics(0)
		screen_h = ctypes.windll.user32.GetSystemMetrics(1)
	except Exception:
		screen_w, screen_h = 1280, 720

	max_w = max(320, screen_w - margin)
	max_h = max(240, screen_h - margin)

	h, w = image.shape[:2]
	scale = min(max_w / w, max_h / h, 1.0)

	if scale < 1.0:
		new_w = int(w * scale)
		new_h = int(h * scale)
		return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

	return image


def run_with_solutions_face_mesh(image):
	mp_face_mesh = mp.solutions.face_mesh

	with mp_face_mesh.FaceMesh(
		static_image_mode=True,
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
	) as face_mesh:
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = face_mesh.process(rgb)

		output = image.copy()
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				draw_landmarks(output, face_landmarks.landmark)

		return output


def run_with_tasks_face_landmarker(image):
	model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
	ensure_face_landmarker_model(model_path)

	BaseOptions = mp.tasks.BaseOptions
	FaceLandmarker = mp.tasks.vision.FaceLandmarker
	FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
	VisionRunningMode = mp.tasks.vision.RunningMode

	options = FaceLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=model_path),
		running_mode=VisionRunningMode.IMAGE,
		num_faces=1,
	)

	with FaceLandmarker.create_from_options(options) as landmarker:
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		result = landmarker.detect(mp_image)

		output = image.copy()
		if result.face_landmarks:
			for face_landmarks in result.face_landmarks:
				draw_landmarks(output, face_landmarks)

		return output


def main():
	image_path = os.path.join(
		os.path.dirname(__file__),
		"asian-man-isolated-expressing-emotions.jpg",
	)

	if not os.path.exists(image_path):
		raise RuntimeError(
			"Image file not found: "
			+ image_path
		)

	image = cv2.imread(image_path)
	if image is None:
		raise RuntimeError("Failed to load image: " + image_path)

	try:
		if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
			output = run_with_solutions_face_mesh(image)
		else:
			output = run_with_tasks_face_landmarker(image)
	except Exception as exc:
		raise RuntimeError(
			"Face landmark initialization failed. "
			"If using modern MediaPipe builds, this script uses tasks API fallback. "
			f"Original error: {exc}"
		)

	while True:
		display = resize_to_fit_screen(output)
		cv2.imshow("MediaPipe FaceMesh (468 Landmarks)", display)
		key = cv2.waitKey(30) & 0xFF
		if key == 27:
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
