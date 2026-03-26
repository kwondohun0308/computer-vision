import os

import cv2 as cv
import matplotlib.pyplot as plt


def main() -> None:
	# 현재 스크립트 파일과 같은 폴더에 있는 입력 이미지를 지정합니다.
	# 절대 경로를 하드코딩하지 않아도 되어, 다른 PC에서도 그대로 실행하기 쉽습니다.
	image_path = os.path.join(os.path.dirname(__file__), "mot_color70.jpg")

	# 컬러(BGR) 영상으로 이미지를 읽습니다.
	# OpenCV 기본 채널 순서는 RGB가 아니라 BGR입니다.
	image_bgr = cv.imread(image_path, cv.IMREAD_COLOR)
	# 파일 경로가 잘못되었거나 파일이 손상된 경우 None이 반환되므로 예외 처리합니다.
	if image_bgr is None:
		raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

	# SIFT는 보통 그레이스케일 입력에서 특징을 검출하므로 색상 정보를 제거합니다.
	image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

	# SIFT 객체를 생성합니다.
	# nfeatures는 검출할 특징점 최대 개수를 제어하며,
	# 값이 작을수록 계산량은 줄고 중요한 점 위주로 남게 됩니다.
	sift = cv.SIFT_create(nfeatures=400)
	# detectAndCompute():
	# 1) keypoints: 특징점 위치/크기/방향 정보
	# 2) descriptors: 각 특징점을 수치 벡터(128차원)로 표현한 기술자
	keypoints, descriptors = sift.detectAndCompute(image_gray, None)

	# 특징점을 원본 이미지 위에 그립니다.
	# DRAW_RICH_KEYPOINTS 플래그를 사용하면 원의 크기(스케일)와 방향도 함께 표시됩니다.
	keypoint_image = cv.drawKeypoints(
		image_bgr,
		keypoints,
		None,
		flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
	)

	# matplotlib은 RGB 순서를 사용하므로 BGR -> RGB 변환이 필요합니다.
	image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
	keypoint_image_rgb = cv.cvtColor(keypoint_image, cv.COLOR_BGR2RGB)

	# 특징점 통계를 출력해 파라미터 조정(nfeatures 등)에 활용합니다.
	print(f"검출된 특징점 개수: {len(keypoints)}")
	if descriptors is not None:
		print(f"디스크립터 shape: {descriptors.shape}")

	# 원본과 결과를 1행 2열로 나란히 시각화합니다.
	plt.figure(figsize=(14, 6))

	plt.subplot(1, 2, 1)
	plt.imshow(image_rgb)
	plt.title("Original Image")
	plt.axis("off")

	plt.subplot(1, 2, 2)
	plt.imshow(keypoint_image_rgb)
	plt.title("SIFT Keypoints")
	plt.axis("off")

	# 서브플롯 간격 자동 조정 후 출력합니다.
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# 직접 실행한 경우에만 main()을 호출합니다.
	main()
