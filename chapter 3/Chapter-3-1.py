import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ===== 이미지 불러오기 =====
# edgeDetectionImage 파일을 읽어옵니다
image = cv.imread('edgeDetectionImage.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# ===== 그레이스케일 변환 =====
# 컬러 이미지를 흑백(그레이스케일)으로 변환합니다
# cv.COLOR_BGR2GRAY: BGR 형식의 컬러를 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# ===== Sobel 필터를 사용한 에지 검출 =====
# X축 방향의 에지 검출
# cv.CV_64F: 64비트 부동소수점 형식
# 1, 0: X 방향 미분 (1차 미분)
# ksize=5: 5x5 커널 사용
sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=5)

# Y축 방향의 에지 검출
# 0, 1: Y 방향 미분 (1차 미분)
sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=5)

# ===== 에지 강도 계산 =====
# X와 Y 방향의 에지를 결합하여 전체 에지 강도를 계산합니다
# Magnitude: sqrt(sobel_x^2 + sobel_y^2)
edge_magnitude = cv.magnitude(sobel_x, sobel_y)

# ===== uint8 형식으로 변환 =====
# 부동소수점 값을 0~255 범위의 정수로 정규화합니다
# 이미지 표시에 필요한 표준 형식입니다
edge_intensity = cv.convertScaleAbs(edge_magnitude)

# ===== 결과 시각화 =====
# 2개의 이미지를 나란히 표시합니다
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 원본 이미지 (그레이스케일) 표시
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original image (Grayscale)', fontsize=12, fontweight='bold')
axes[0].axis('off')  # 좌표축 제거

# 에지 검출 결과 표시
axes[1].imshow(edge_intensity, cmap='gray')
axes[1].set_title('Sobel Edge Detection Result', fontsize=12, fontweight='bold')
axes[1].axis('off')  # 좌표축 제거

# 전체 제목 설정
plt.suptitle('Sobel Edge Detection Result', fontsize=14, fontweight='bold', y=0.95)

# 레이아웃 자동 조정
plt.tight_layout()

# 결과 이미지 저장 (선택사항)
cv.imwrite('sobel_edge_result.jpg', edge_intensity)
print("에지 검출 결과가 'sobel_edge_result.jpg'로 저장되었습니다.")

# 그래프 표시
plt.show()
