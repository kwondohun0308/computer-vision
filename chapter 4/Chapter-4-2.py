import os

import cv2 as cv
import matplotlib.pyplot as plt


def main() -> None:
	# 두 입력 영상의 경로를 현재 파일 기준으로 구성합니다.
	base_dir = os.path.dirname(__file__)
	image1_path = os.path.join(base_dir, "mot_color70.jpg")
	image2_path = os.path.join(base_dir, "mot_color83.jpg")

	# 두 영상을 컬러(BGR)로 읽습니다.
	image1 = cv.imread(image1_path, cv.IMREAD_COLOR)
	image2 = cv.imread(image2_path, cv.IMREAD_COLOR)

	# 파일 로드 실패 시 즉시 원인을 알 수 있도록 예외 처리합니다.
	if image1 is None:
		raise FileNotFoundError(f"첫 번째 이미지를 찾을 수 없습니다: {image1_path}")
	if image2 is None:
		raise FileNotFoundError(f"두 번째 이미지를 찾을 수 없습니다: {image2_path}")

	# SIFT 특징 추출을 위해 그레이스케일로 변환합니다.
	gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

	# SIFT 특징점/디스크립터를 계산합니다.
	# nfeatures를 조절하면 검출 개수(및 속도/정확도 균형)를 조정할 수 있습니다.
	sift = cv.SIFT_create(nfeatures=500)
	keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
	keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

	# 특징이 거의 없는 영상에서는 디스크립터가 None일 수 있으므로 방어 코드가 필요합니다.
	if descriptors1 is None or descriptors2 is None:
		raise RuntimeError("SIFT 디스크립터를 계산하지 못했습니다.")

	# BFMatcher:
	# SIFT 디스크립터는 float 기반이므로 L2 거리(norm)를 사용합니다.
	# crossCheck=False로 두 최근접 이웃을 구해 ratio test를 적용할 수 있게 합니다.
	bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
	# 각 특징점에 대해 가장 가까운 이웃 2개를 찾습니다.
	knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

	# Lowe ratio test:
	# 최선 매칭(m)이 차선 매칭(n)보다 충분히 더 가깝다면 좋은 매칭으로 채택합니다.
	# threshold가 작을수록 더 엄격(정확도↑, 개수↓), 클수록 느슨(개수↑, 오매칭↑)합니다.
	ratio_thresh = 0.75
	good_matches = []
	for m, n in knn_matches:
		if m.distance < ratio_thresh * n.distance:
			good_matches.append(m)

	# 거리 기준 정렬 후 상위 일부만 그려서 시각적 혼잡을 줄입니다.
	good_matches = sorted(good_matches, key=lambda x: x.distance)
	max_draw = 80
	draw_matches = good_matches[:max_draw]

	# drawMatches로 두 영상을 한 캔버스에 이어 붙이고 대응선을 표시합니다.
	# NOT_DRAW_SINGLE_POINTS 플래그는 매칭되지 않은 키포인트는 생략합니다.
	match_image = cv.drawMatches(
		image1,
		keypoints1,
		image2,
		keypoints2,
		draw_matches,
		None,
		flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	)

	# matplotlib 표시에 맞춰 BGR -> RGB 변환합니다.
	match_image_rgb = cv.cvtColor(match_image, cv.COLOR_BGR2RGB)

	# 결과 통계를 출력해 매칭 품질을 빠르게 확인합니다.
	print(f"이미지1 특징점 개수: {len(keypoints1)}")
	print(f"이미지2 특징점 개수: {len(keypoints2)}")
	print(f"ratio test 통과 매칭 개수: {len(good_matches)}")
	print(f"시각화된 매칭 개수: {len(draw_matches)}")

	# 매칭 결과를 단일 화면으로 출력합니다.
	plt.figure(figsize=(16, 8))
	plt.imshow(match_image_rgb)
	plt.title("SIFT Feature Matching (BFMatcher + Ratio Test)")
	plt.axis("off")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# 스크립트를 직접 실행할 때 진입점 역할을 합니다.
	main()
