# chapter 3
## 과제1 설명 및 요구사항 (체크보드 기반 카메라 캘리브레이션)
 - edgeDetectionImage 이미지를 그레이 스케일로 변환
 - Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출
 - 검출된 에지 강도 이미지를 시각화
 - cv.imread()를 사용하여 이미지를 불러옴
 - cv.cvtColor()를 사용하여 그레이스케일로 변환
 - cvSobel()을 사용하여 x축(cv.CV_64F, 1, 0)과 y축(cv.CV_64F, 0,1) 방향의 에지를 검출
 - cv.magnitude()를 사용하여 에지 강도 계산
 - Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화

과제 한줄 요약 - OpenCV의 소벨(Sobel) 필터로 이미지의 가로·세로 밝기 변화(미분)를 계산하여, 물체의 선명한 윤곽선(에지)을 찾아내는 과정

<details>
	<summary>과제 1 전체 코드</summary>
	
		import cv2 as cv
		import numpy as np
		import matplotlib.pyplot as plt
		
		image = cv.imread('edgeDetectionImage.jpg')
		
		if image is None:
		    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
		    exit()
		
		gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		
		sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=5)
		
		sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=5)
		
		edge_magnitude = cv.magnitude(sobel_x, sobel_y)
		
		edge_intensity = cv.convertScaleAbs(edge_magnitude)
		
		fig, axes = plt.subplots(1, 2, figsize=(12, 5))
		
		axes[0].imshow(gray_image, cmap='gray')
		axes[0].set_title('Original image (Grayscale)', fontsize=12, fontweight='bold')
		axes[0].axis('off')
		
		axes[1].imshow(edge_intensity, cmap='gray')
		axes[1].set_title('Sobel Edge Detection Result', fontsize=12, fontweight='bold')
		axes[1].axis('off')
		
		plt.suptitle('Sobel Edge Detection Result', fontsize=14, fontweight='bold', y=0.95)
		
		plt.tight_layout()
		
		cv.imwrite('sobel_edge_result.jpg', edge_intensity)
		print("에지 검출 결과가 'sobel_edge_result.jpg'로 저장되었습니다.")
		
		plt.show()

</details>

<img width="1207" height="571" alt="화면 캡처 2026-03-19 152828" src="https://github.com/user-attachments/assets/add9d531-1efd-43ae-a170-78238b48bd0f" />


## 과제 1 주요 코드 설명
1. 소벨 필터를 이용한 미분
   에지는 이미지에서 색상이나 밝기가 급격하게 변하는 부분이며 미분을 통해 찾을 수 있습니다.

	```python
		sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=5)
		
		sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=5)
	```
	sobel_x - 가로 방향의 픽셀 값 변화를 계산하고, 이를 통해 세로 방향의 에지가 두드러지게 검출됩니다. (1,0은 x축 방향으로 1차 미분, y축 방향은 미분하지 않음을 의미)
	
	sobel_y - 세로 방향의 픽셀 값 변화를 계산하고 이를 통해 가로 방향의 에지가 두드러지게 검출됩니다. (0,1은 y축 방향으로 1차 미분, x축 방향은 미분하지 않음을 의미)
	
	cv.CV_64F - 미분을 하면 밝기 값이 음수가 될 수도 있으므로 64비트 실수형으로 저장합니다.

	ksize=5 - 에지를 찾을 때 5*5 크기의 마스크를 사용한다는 뜻

	<details><summary>보충설명</summary>
	이미지를 왼쪽에서 오른쪽(가로)으로 훑으면서 밝기 변화를 찾습니다. 가로로 이동하다가 밝기가 확 변했다는 것은, 그 자리에 세로 방향으로 길게 뻗은 경계선(수직선)이 존재한다는 뜻입니다.

	마찬가지로 이미지를 위에서 아래(세로)로 훑으면서 밝기 변화를 찾습니다. 위에서 아래로 떨어지다가 밝기가 확 변했다면, 바닥처럼 가로 방향으로 길게 뻗은 경계선(수평선)을 밟았다는 뜻입니다.
	</details>

2. 결과 합성 및 이미지화
   ```python

		edge_magnitude = cv.magnitude(sobel_x, sobel_y)
		
		edge_intensity = cv.convertScaleAbs(edge_magnitude)
   ```
   cv.magnitude - 앞서 구한 x축 변화량과 y축 변화량을 합치는 과정입니다. 피타고라스 정리를 이용해서 전체적인 에지의 강도를 구합니다.

   edge_intensity - 앞의 CV_64F라는 실수형으로 계산된 데이터를 눈으로 볼수 있는 이미지 형식으로 되돌려주며, 음수는 절댓값을 취해 양수로 만들어 에지 강도로 표현합니다.


## 과제2 설명 및 요구사항 (캐니 에지 및 허프 변환을 이용한 직선 검출)
 - dabo 이미지에 캐니 에지 검출을 사용하여 에지 맵 생성
 - 허프 변환을 사용하여 이미지에 직선 검출
 - 검출된 직선을 원본 이미지에서 빨간색으로 표시
 - cv.Canny()를 사용하여 에지 맵 생성
 - cv.HoughtLinesP()를 사용하여 직선 검출
 - cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림
 - Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화

과제 한줄 요약 - 캐니(Canny) 에지 검출로 이미지의 윤곽선을 먼저 찾고, 허프 변환(Hough Transform)을 이용해 그 안에서 의미 있는 '직선' 성분만 추출하여 시각화하는 과정

<details>
	<summary>과제 2 전체 코드</summary>
	
		import cv2 as cv
		import numpy as np
		import matplotlib.pyplot as plt
		
		image = cv.imread('dabo.jpg')
		
		if image is None:
		    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
		    exit()
		
		gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		
		edges = cv.Canny(gray_image, 100, 200)
		
		lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
		
		image_with_lines = image.copy()
		
		if lines is not None:
		    for line in lines:
		        x1, y1, x2, y2 = line[0]
		        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
		
		    print(f"검출된 직선의 개수: {len(lines)}개")
		else:
		    print("검출된 직선이 없습니다. 파라미터를 조정해주세요.")
		
		fig, axes = plt.subplots(1, 3, figsize=(18, 5))
		
		axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
		axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
		axes[0].axis('off')
		
		axes[1].imshow(edges, cmap='gray')
		axes[1].set_title('Canny Edge Detection', fontsize=12, fontweight='bold')
		axes[1].axis('off')
		
		axes[2].imshow(cv.cvtColor(image_with_lines, cv.COLOR_BGR2RGB))
		axes[2].set_title('Hough Line Transform', fontsize=12, fontweight='bold')
		axes[2].axis('off')
		
		plt.suptitle('Canny Edge Detection & Hough Line Transform', fontsize=14, fontweight='bold', y=1.00)
		
		plt.tight_layout()
		
		cv.imwrite('hough_lines_result.jpg', image_with_lines)
		print("직선 검출 결과가 'hough_lines_result.jpg'로 저장되었습니다.")
		
		plt.show()


</details>

<img width="1806" height="572" alt="화면 캡처 2026-03-19 154009" src="https://github.com/user-attachments/assets/a2323e33-5ed2-4893-8880-abe3a48f58b6" />

## 과제 2 주요 코드 설명

1. 캐니 에지 검출
	```python
   		edges = cv.Canny(gray_image, 100, 200)
 	```
	100, 200은 이중 임계값이라고 불리며 Canny 알고리즘에서 권장하는 2:1 비율을 적용한 것입니다. 200 이상은 확실한 에지(경계선)이고, 100 미만은 노이즈로 판별, 100 ~ 200 사이는 애매하지만 200이 넘는 확실한 에지와 선으로 이어져있다면 에지로 판별됩니다. 이 방식으로 선이 중간에 끊기지 않고 자연스럽게 이어지는 윤곽선을 얻을수 있습니다.

2. 확률적 허프 변환
	```python
 		lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
 	```
	1, np.pi/180 - 거리는 1픽셀 단위로, 각도는 1도(π/180라디안)단위로 아주 촘촘하게 검사하겠다는 뜻

	50 - 최소 50개의 점이 투표해야 이것을 하나의 직선으로 인정하겠다는 뜻
   
   	minLineLength=50 - 최소 직선 길이입니다. 아무리 직선같아보여도 길이가 50픽셀도 되지 않는것은 무시한다는 뜻
   
   	maxLineGap=10 - 최대 선분 간격입니다. 점선이나 살짝 지워진 선처럼 중간에 끊겨 있어도, 그 틈이 10픽세 이내라면 원래 하나의 직선으로 인식

4. 검출된 직선 그리기
	```python
 		if lines is not None:
		    for line in lines:
		        x1, y1, x2, y2 = line[0]
		        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
		
		    print(f"검출된 직선의 개수: {len(lines)}개")
		else:
		    print("검출된 직선이 없습니다. 파라미터를 조정해주세요.")
 	```
	허프 변환에서 인식된 직선들은 [시작점 x, 시작점 y, 끝점 x, 끝점 y] 라는 4개의 좌표를 가져옵니다. 이 좌표를 이용해 원본 이미지 위에 (0,0,255) 즉, 빨간색으로 두께 2 직선을 그어줍니다.

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
	
		import cv2 as cv
		import numpy as np
		import matplotlib.pyplot as plt
		
		image = cv.imread('coffee cup.jpg')
		
		if image is None:
		    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
		    exit()
		
		mask = np.zeros(image.shape[:2], np.uint8)
		
		bgdModel = np.zeros((1, 65), np.float64)
		fgdModel = np.zeros((1, 65), np.float64)
		
		h, w = image.shape[:2]
		x = int(w * 0.05)
		y = int(h * 0.05)
		width = int(w * 0.90)
		height = int(h * 0.90)
		rect = (x, y, width, height)
		
		print(f"ROI 정보: x={x}, y={y}, width={width}, height={height}")
		
		cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
		
		output_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
		
		image_no_bg = image.copy()
		
		mask_3channel = cv.cvtColor(output_mask, cv.COLOR_GRAY2BGR)
		
		image_no_bg = cv.bitwise_and(image_no_bg, mask_3channel)
		
		fig, axes = plt.subplots(1, 3, figsize=(18, 5))
		
		image_with_rect = image.copy()
		cv.rectangle(image_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
		axes[0].imshow(cv.cvtColor(image_with_rect, cv.COLOR_BGR2RGB))
		axes[0].set_title('Original image (Initial region)', fontsize=12, fontweight='bold')
		axes[0].axis('off')
		
		axes[1].imshow(output_mask, cmap='gray')
		axes[1].set_title('GrabCut mask (object = white)', fontsize=12, fontweight='bold')
		axes[1].axis('off')
		
		axes[2].imshow(cv.cvtColor(image_no_bg, cv.COLOR_BGR2RGB))
		axes[2].set_title('Background removal result (object only)', fontsize=12, fontweight='bold')
		axes[2].axis('off')
		
		plt.suptitle('GrabCut Object Extraction', fontsize=14, fontweight='bold', y=1.00)
		
		plt.tight_layout()
		
		cv.imwrite('grabcut_mask.jpg', output_mask)
		cv.imwrite('grabcut_result.jpg', image_no_bg)
		print("마스크가 'grabcut_mask.jpg'로 저장되었습니다.")
		print("추출 결과가 'grabcut_result.jpg'로 저장되었습니다.")
		
		plt.show()

</details>

<img width="1803" height="565" alt="화면 캡처 2026-03-19 153947" src="https://github.com/user-attachments/assets/7d9142ea-849e-48c6-8311-d3bd03512ea6" />

## 과제 3 주요 코드 설명

1. 알고리즘 세팅
	```python
 		mask = np.zeros(image.shape[:2], np.uint8)
		
		bgdModel = np.zeros((1, 65), np.float64)
		fgdModel = np.zeros((1, 65), np.float64)
 	```
	mask - 최종적으로 어디가 객체이고 어디가 배경인지 결과를 그려넣을 배열

	bgdModel, fgdModel: 컴퓨터가 분석한 색상 분포 모델(가우시안 혼합 모델)을 임시로 적어두는 내부 메모장입니다. 빈 공간으로 던져주면 GrabCut 함수가 알아서 채워가며 학습합니다.

2. 초기 가이드라인 설정
	```python
		h, w = image.shape[:2]
		x = int(w * 0.05)
		y = int(h * 0.05)
		width = int(w * 0.90)
		height = int(h * 0.90)
		rect = (x, y, width, height)
 	```
   사각형을 하나 그릴테니 이 사각형 바깥은 무조건 배경이고 안쪽에 찾아야할 객체가 있다고 알려주는 과정
   사진 크기의 바깥쪽 5%를 제외하고 안쪽 90%를 채우는 사각형을 만들었습니다.

3. GrabCut 실행
	```python
		cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
	```
	5 - 반복횟수이며 컴퓨터가 전경과 배경의 색상 차이를 비교하며 경계선을 다듬는 과정을 5번 반복하며 반복횟수가 높을수록 정교해집니다.
	
	cv.GC_INIT_WITH_RECT - 위에서 만든 사각형을 기준으로 초기화를 하라는 명령어입니다. 사각형의 바깥부분의 픽셀 색상들을 모아서 확실한 배경색으로 규정하고 사각형 안쪽에서 그 배경색과 다른 색상들을 찾아내어 객체로 분리합니다.

4. 마스크 결과물 해석
	```python
	output_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
 	```
 	cv.GC_BGD - 확실한 배경
	cv.GC_FGD - 확실한 객체
	cv.GC_PR_BGD - 배경 의심
	cv.GC_PR_FGD - 객체 의심

	4가지 중에서 확실한 객체와 객체 의심을 하얀색으로 칠하고, 나머지 배경을 검은색으로 덮으라는 명령어

5. 배경 날리기
	```python
 		mask_3channel = cv.cvtColor(output_mask, cv.COLOR_GRAY2BGR)
		
		image_no_bg = cv.bitwise_and(image_no_bg, mask_3channel)
	```
   만들어진 흑백 마스크를 원본 사진에 덮어 배경을 오리는 과정
   mask_3channel - 원본 이미지는 컬러인데 마스크는 흑백이므로 3채널로 맞춰줌
   
   cv.bitwise_and - 원본 이미지와 마스크를 and연산하고 마스크에서 흰색부분은 그대로 원본색상이 살아남고, 검은색 부분은 원본에 곱해져서 지워집니다.
