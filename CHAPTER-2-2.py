import cv2

img = cv2.imread("rose.png")
if img is None:
	raise FileNotFoundError("사진 파일을 찾을 수 없습니다.")

# 이미지의 높이(h), 너비(w)를 구한다.
# shape[:2]는 (height, width)를 의미한다.
h, w = img.shape[:2]

# 회전 기준점은 이미지의 중심으로 잡는다.
# 중심 기준으로 회전하면 물체가 화면 중앙을 축으로 자연스럽게 회전한다.
center = (w / 2, h / 2)

# getRotationMatrix2D는 2x3 아핀 변환 행렬을 생성한다.
# 이미지 중심 기준 +30도 회전
# 동시에 전체 크기를 0.8배로 축소

M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 회전/스케일 행렬에 평행이동 성분을 추가한다.
# 아핀 변환 행렬의 마지막 열은 각각 x, y 방향 이동량을 의미한다.
# 따라서 아래 코드는 "회전+축소" 결과를 다시
# x축으로 +80픽셀, y축으로 -40픽셀 이동시키는 의미가 된다.
M[0, 2] += 80
M[1, 2] -= 40

# warpAffine은 입력 이미지에 2x3 변환 행렬을 적용해 새로운 이미지를 만든다.
# 출력 크기는 원본과 동일하게 (w, h)로 유지한다.
transformed = cv2.warpAffine(img, M, (w, h))

# 원본 이미지와 변환 결과를 각각 창으로 띄워 비교한다.
cv2.imshow("Original", img)
cv2.imshow("Transformed", transformed)

# 아무 키나 누를 때까지 창을 유지한다.
cv2.waitKey(0)

# 열린 창을 모두 닫고 프로그램을 종료한다.
cv2.destroyAllWindows()
