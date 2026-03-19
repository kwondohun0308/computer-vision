import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ===== 이미지 불러오기 =====
# dabo 이미지를 읽어옵니다
image = cv.imread('dabo.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# ===== 그레이스케일 변환 =====
# 에지 검출을 위해 컬러 이미지를 흑백으로 변환합니다
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# ===== 캐니 에지 검출 =====
# cv.Canny(): 이미지에서 에지를 검출합니다
# threshold1=100: 낮은 임계값 (약한 에지)
# threshold2=200: 높은 임계값 (강한 에지)
# 두 값 사이의 에지는 강한 에지와 연결되었을 때만 포함됩니다
edges = cv.Canny(gray_image, 100, 200)

# ===== 허프 변환을 사용한 직선 검출 =====
# cv.HoughLinesP(): 확률적 허프 변환으로 직선을 검출합니다
# rho=1: 해상도 1 픽셀
# theta=np.pi/180: 각도 해상도 1도
# threshold=50: 최소 투표 수 (값이 크면 더 명확한 직선만 검출)
# minLineLength=50: 최소 직선 길이 50 픽셀
# maxLineGap=10: 최대 선분 간격 10 픽셀 (이 정도 간격이면 같은 직선으로 봄)
lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

# ===== 검출된 직선을 원본 이미지에 그리기 =====
# 직선이 검출되었는지 확인
image_with_lines = image.copy()  # 원본 이미지 복사 (원본 보존)

if lines is not None:
    # 검출된 각 직선을 반복하며 그립니다
    for line in lines:
        # 직선의 시작점과 끝점 좌표를 추출합니다
        x1, y1, x2, y2 = line[0]
        
        # 직선을 그립니다
        # cv.line(이미지, 시작점, 끝점, 색상, 두께)
        # (0, 0, 255): BGR 형식의 빨간색
        # 2: 선의 두께 2픽셀
        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(f"검출된 직선의 개수: {len(lines)}개")
else:
    print("검출된 직선이 없습니다. 파라미터를 조정해주세요.")

# ===== 결과 시각화 =====
# 3개의 이미지를 나란히 표시합니다
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 원본 이미지 표시
axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# 캐니 에지 검출 결과 표시
axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Canny Edge Detection', fontsize=12, fontweight='bold')
axes[1].axis('off')

# 직선이 그려진 이미지 표시
axes[2].imshow(cv.cvtColor(image_with_lines, cv.COLOR_BGR2RGB))
axes[2].set_title('Hough Line Transform', fontsize=12, fontweight='bold')
axes[2].axis('off')

# 전체 제목 설정
plt.suptitle('Canny Edge Detection & Hough Line Transform', fontsize=14, fontweight='bold', y=1.00)

# 레이아웃 자동 조정
plt.tight_layout()

# 결과 이미지 저장 (선택사항)
cv.imwrite('hough_lines_result.jpg', image_with_lines)
print("직선 검출 결과가 'hough_lines_result.jpg'로 저장되었습니다.")

# 그래프 표시
plt.show()
