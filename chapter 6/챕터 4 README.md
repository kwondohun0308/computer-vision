# chapter 4
## 과제1 설명 및 요구사항 (SIFT를 이용한 특징점 검출 및 시각화)
 - 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT 알고리즘을 사용하여 특징점을 검출하고 이를 시각화
 - cv.SIFT_create()를 사용하여 SIFT 객체를 생성
 - detectAndCompute()를 사용하여 특징점을 검출
 - cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
 - matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력

과제 한줄 요약 - SIFT 알고리즘을 적용하여 이미지의 스케일(크기) 및 회전 변화에 강건한(Robust) 고유 특징점(Keypoint)을 검출하고, 이를 128차원 디스크립터(Descriptor)로 추출하여 시각화하는 과정

<details>
	<summary>과제 1 전체 코드</summary>
	
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


</details>

<img width="1402" height="667" alt="1번 과제" src="https://github.com/user-attachments/assets/3af375b6-5fc6-4a78-ab52-32093d12adf6" />



## 과제 1 주요 코드 설명
1. SIFT 객체 초기화 및 연산 최적화

	```python
		sift = cv.SIFT_create(nfeatures=400)
	```
	nfeatures=400 파라미터는 검출할 특징점의 최대 개수를 제한합니다.

	이미지 내에는 수많은 코너와 에지가 존재하지만 연산 효율성과 향후 매칭 과정에서 병목현상을 방지하기 위해 대비가 가장 높고 안정적인 상위 400개의 극점만 추출하도록 제한함. 이를 통해 노이즈에 해당하는 무의미한 특징점들을 사전에 필터링 가능


2. 특징점 검출 및 디스크립터(Descriptor) 계산
   ```python
		keypoints, descriptors = sift.detectAndCompute(image_gray, None)
   ```
   SIFT의 핵심인 기하학적 특징 검출(detect)과 특징 벡터 추출(compute)을 동시에 수행
   용어 설명
   keypoints - 특징점의 상태정보이며 이 특징점이 얼마나 넓은 영역에서 검출되었는지, 그리고 어느 방향을 가리키고 있는지를 함께 기억함
   
   descriptors - 컴퓨터는 이미지를 숫자 배열로 인식하기 때문에 특징점 주변 영역의 명암 패턴을 컴퓨터가 이해하기 쉽게 128개의 숫자로 요약해둔것

3. 메타데이터 시각화
   ```python
		keypoint_image = cv.drawKeypoints(
			image_bgr,
			keypoints,	
			None,
			flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,	
		)			
   ```
   추출된 특징점 데이터를 원본 이미지 상에 매핑하여 시각적으로 검증하는 단계
   
   DRAW_RICH_KEYPOINTS 플래그를 적용함으로써, 단순한 점의 위치를 넘어 특징점이 내포한 메타데이터를 렌더링합니다. 시각화된 원의 반지름은 검출된 특징점의 스케일(영역의 크기)을 의미하며, 원 내부의 직선은 해당 특징점의 주변 그래디언트를 통해 도출된 주방향(Orientation)을 나타냅니다.

## 과제2 설명 및 요구사항 (SIFT를 이용한 두 영상 간 특징점 매칭)
 - 두 개의 이미지(mot_color70.jpg, mot_color83.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화
 - cv.imread()를 사용하여 두 개의 이미지를 불러옴
 - cv.SIFT_create()를 사용하여 특징점을 추출
 - cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭
 - cv.drawMatches()를 사용하여 매칭 결과를 시각화
 - matplotlib을 이용하여 매칭 결과를 출력

과제 한줄 요약 - SIFT로 추출한 두 이미지의 특징 벡터(바코드)들을 전수조사 방식으로 비교하고, 엄격한 '비율 검사(Ratio Test)'를 통과한 확실한 짝만 선별하여 매칭하는 과정

<details>
	<summary>과제 2 전체 코드</summary>
	
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



</details>

<img width="1604" height="864" alt="2번 과제" src="https://github.com/user-attachments/assets/0edfe485-ec3f-40ca-84fc-fc369f556329" />


## 과제 2 주요 코드 설명

1. 전수조사 방식의 매칭기 준비 (BFMatcher)
	```python
   		bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
 	```
	BF(Brute-Force) - 첫 번째 이미지의 특징점 1개당, 두 번째 이미지의 모든 특징점을 일일이 다 대조해 보면서 가장 비슷한 녀석을 찾습니다.

	cv.NORM_L2 - 128차원으로 이루어진 두 바코드 사이의 직선거리를 계산하라는 뜻입니다. 거리가 짧을수록 두 바코드가 비슷하게 생겼다는 뜻

2. 가장 닮은 후보 2개 찾기 (k-NN Match)
	```python
 		knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
 	```
	가장 비슷한 1개만 찾는 것이 아니라 k=2를 설정하여 최선과 차선 두개의 후보를 뽑아옵니다.

3. 데이비드 로우의 비율 검사
	```python
			ratio_thresh = 0.75
			good_matches = []
			for m, n in knn_matches:
				if m.distance < ratio_thresh * n.distance:
					good_matches.append(m)
 	```
	반복되는 패턴의 경우에 특징점이 너무 흔하게 생겨서 다른곳과 매칭되는 현상이 발생하게 됩니다.
	이로 인해 방금 뽑아온 최선과 차선의 거리를 비교합니다. 만약 최선의 거리가 압도적으로 가깝다면 진짜일 것이고,
	둘의 거리가 비슷하다면 흔한 패턴일 확률이 높으므로 과감하게 버립니다.
	0.75로 설정한 이유는 차선 후보와 최소 25% 이상의 격차가 나야만 진짜로 인정하겠다는 뜻으로 설정했습니다.

4. 결과 시각화 (선 그리기)
	```python
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
 	```
	합격한 good_matches들은 거리가 짧은 순서대로 정렬한 뒤, 화면이 너무 지저분해지는것을 방지하기 위해 상위 80개만 끊어서 선으로 연결함.
	NOT_DRAW_SINGLE_POINTS 옵션을 통해 짝을 찾지 못한 점들은 화면에 그리지 않고 깔끔하게 생략합니다.

## 과제3 설명 및 요구사항 (GrabCut을 이용한 대화식 영역 분할 및 객체 추출)
 - coffee cup 이미지로 사용하가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체 추출
 - 객체 추출 결과를 마스크 형태로 시각화
 - 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력
 - cv.grabCut()를 사용하여 대화식 분할을 수행
 - 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
 - 마스크를 사용하여 원본 이미지에서 배경을 제거
 - matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화

과제 한줄 요약 - 사용자가 피사체 주변에 대략적인 사각형 영역만 지정해주면, GrabCut 알고리즘이 스스로 색상 분포를 분석해 배경과 전경(객체)을 분리하여 원하는 피사체만 깔끔하게 오려내는(누끼 따기) 과정

<details>
	<summary>과제 3 전체 코드</summary>
	
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

</details>

<img width="1801" height="765" alt="3번 과제" src="https://github.com/user-attachments/assets/d0386bc4-ed3d-4dd0-b22d-777a8426568d" />


## 과제 3 주요 코드 설명

1. 호모그래피(Homography) 행렬 계산과 RANSAC
	```python
 		homography, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
			if homography is None:
				raise RuntimeError("호모그래피 행렬 계산에 실패했습니다.")
 	```
	cv.findHomography는 두 이미지의 시점을 맞추는 변환 행렬을 계산할 때, 잘못된 매칭점(Outlier)의 간섭을 막기 위해 RANSAC 알고리즘을 사용합니다.

	무작위로 4쌍의 점만 뽑아 임시 모델을 세우고 나머지 점들을 대입해 정상점(Inlier)을 판별하는 과정을 무수히 반복하며, 최종적으로 가장 많은 정상점이 지지하는 최적의 모델을 채택하여 정교한 파노라마 합성을 완성합니다.

2. 캔버스 크기 계산 및 확장 (Bounding Box)
	```python
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
 	```
   사진을 기울이고 늘렸을 때 모서리 좌표가 어떻게 변하는지 미리 계산해 보고, 좌표가 마이너스(-)로 넘어가 이미지가 잘려 나가는 것을 방지하기 위해 전체 캔버스 크기를 확장합니다.

3. 이미지 워핑(Warping) 및 합성
	```python
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
	```
	translation @ homography: 파이썬에서 @는 행렬의 곱셈을 의미합니다.

	이제 커다란 캔버스(aligned_canvas) 위에 비틀고 이동시킨 첫 번째 사진이 넓게 깔려있고 기준이 되는 두 번째 사진(image2)을 아까 밀어냈던 좌표(y_offset, x_offset)에 정확히 맞춰서 그 위에 덮어씌웁니다.
