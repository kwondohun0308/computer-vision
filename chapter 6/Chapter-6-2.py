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
        max_num_faces=1,             # 최대 찾을 얼굴 개수
        refine_landmarks=True,       # 눈동자나 입술 주변을 더 정밀하게 찾을지 여부
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
