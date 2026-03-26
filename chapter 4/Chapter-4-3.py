import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
	# 정합할 두 이미지를 현재 스크립트 기준 경로에서 불러옵니다.
	base_dir = os.path.dirname(__file__)
	image1_path = os.path.join(base_dir, "img1.jpg")
	image2_path = os.path.join(base_dir, "img2.jpg")

	# image1: 변환(warp)할 소스 영상, image2: 정렬의 기준이 되는 타깃 영상
	image1 = cv.imread(image1_path, cv.IMREAD_COLOR)
	image2 = cv.imread(image2_path, cv.IMREAD_COLOR)

	# 입력 파일 유효성 검사
	if image1 is None:
		raise FileNotFoundError(f"첫 번째 이미지를 찾을 수 없습니다: {image1_path}")
	if image2 is None:
		raise FileNotFoundError(f"두 번째 이미지를 찾을 수 없습니다: {image2_path}")

	# SIFT 특징 추출을 위해 두 영상을 그레이스케일로 변환합니다.
	gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

	# SIFT 특징점/디스크립터 계산
	sift = cv.SIFT_create(nfeatures=1200)
	keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
	keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

	# 특징이 충분하지 않으면 이후 매칭/호모그래피 계산이 불가능합니다.
	if descriptors1 is None or descriptors2 is None:
		raise RuntimeError("SIFT 디스크립터 계산에 실패했습니다.")

	# BFMatcher + KNN(최근접 2개)으로 후보 매칭을 구합니다.
	bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
	knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

	# Lowe ratio test로 오매칭(outlier) 후보를 1차적으로 제거합니다.
	ratio_thresh = 0.7
	good_matches = []
	for pair in knn_matches:
		if len(pair) < 2:
			continue
		m, n = pair
		if m.distance < ratio_thresh * n.distance:
			good_matches.append(m)

	# 호모그래피는 최소 4쌍의 대응점이 필요합니다.
	if len(good_matches) < 4:
		raise RuntimeError(
			f"호모그래피 계산에 필요한 매칭점이 부족합니다. good_matches={len(good_matches)}"
		)

	# 좋은 매칭에서 좌표쌍을 추출합니다.
	# src_pts: image1 좌표, dst_pts: image2 좌표
	src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

	# RANSAC 기반 호모그래피 추정:
	# 이상점 영향을 줄여 더 안정적인 투영 변환 행렬(H)을 계산합니다.
	# inlier_mask는 각 매칭점이 최종 모델에 채택(inlier=1)되었는지 나타냅니다.
	homography, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
	if homography is None:
		raise RuntimeError("호모그래피 행렬 계산에 실패했습니다.")

	# 두 이미지의 코너 좌표를 이용해, 워핑 후 전체가 들어가는 캔버스 크기를 계산합니다.
	h1, w1 = image1.shape[:2]
	h2, w2 = image2.shape[:2]
	corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
	corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
	# image1 코너를 호모그래피로 변환해 워핑 후 영역을 얻습니다.
	warped_corners1 = cv.perspectiveTransform(corners1, homography)
	all_corners = np.vstack((warped_corners1, corners2))

	# 전체 경계 박스(min/max)를 구하고, 음수 좌표가 생기면 평행이동으로 보정합니다.
	[x_min, y_min] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
	[x_max, y_max] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

	translate_x = -x_min
	translate_y = -y_min
	panorama_width = x_max - x_min
	panorama_height = y_max - y_min

	# translation @ homography:
	# 호모그래피 변환 결과를 캔버스 내부로 옮기기 위해 평행이동을 후처리로 결합합니다.
	translation = np.array(
		[[1, 0, translate_x], [0, 1, translate_y], [0, 0, 1]], dtype=np.float64
	)
	warped_image1 = cv.warpPerspective(
		image1, translation @ homography, (panorama_width, panorama_height)
	)

	# 기준 영상(image2)을 동일 캔버스 좌표계에 배치해 정렬 결과를 만듭니다.
	aligned_canvas = warped_image1.copy()
	x_offset = int(translate_x)
	y_offset = int(translate_y)
	aligned_canvas[y_offset : y_offset + h2, x_offset : x_offset + w2] = image2

	# 매칭 시각화를 위해 거리 기준 정렬 + inlier 여부를 함께 묶어서 관리합니다.
	good_matches_with_mask = list(zip(good_matches, inlier_mask.ravel().tolist()))
	good_matches_with_mask.sort(key=lambda item: item[0].distance)
	max_draw = 80
	draw_pairs = good_matches_with_mask[:max_draw]
	draw_matches = [pair[0] for pair in draw_pairs]

	# drawMatches에 전달할 inlier 마스크(1/0) 목록
	matches_mask = [pair[1] for pair in draw_pairs]

	# 매칭선 시각화:
	# inlier만 강조해 보여주므로 호모그래피 품질을 직관적으로 파악하기 좋습니다.
	matching_result = cv.drawMatches(
		image1,
		keypoints1,
		image2,
		keypoints2,
		draw_matches,
		None,
		matchColor=(0, 255, 0),
		singlePointColor=(255, 0, 0),
		matchesMask=matches_mask,
		flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	)

	# matplotlib 표시를 위한 색상 순서 변환(BGR -> RGB)
	aligned_canvas_rgb = cv.cvtColor(aligned_canvas, cv.COLOR_BGR2RGB)
	matching_result_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)

	# 핵심 지표 출력: 특징점 수, 필터링된 매칭 수, RANSAC inlier 수
	inlier_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
	print(f"검출된 특징점 개수 (img1): {len(keypoints1)}")
	print(f"검출된 특징점 개수 (img2): {len(keypoints2)}")
	print(f"좋은 매칭점 개수: {len(good_matches)}")
	print(f"RANSAC Inlier 개수: {inlier_count}")

	# 좌측: 매칭 결과, 우측: 호모그래피 정합 결과를 나란히 표시합니다.
	plt.figure(figsize=(18, 7))

	plt.subplot(1, 2, 1)
	plt.imshow(matching_result_rgb)
	plt.title("Matching Result (SIFT + BFMatcher + Ratio Test)")
	plt.axis("off")

	plt.subplot(1, 2, 2)
	plt.imshow(aligned_canvas_rgb)
	plt.title("Warped Image (Homography Alignment)")
	plt.axis("off")

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# 스크립트를 단독 실행할 때 진입점
	main()
