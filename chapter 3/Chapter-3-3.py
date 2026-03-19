import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ===== 이미지 불러오기 =====
# coffee cup 이미지를 읽어옵니다
image = cv.imread('coffee cup.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# ===== GrabCut 알고리즘 초기화 =====
# 마스크: 각 픽셀이 배경/전경인지 나타냅니다
# 초기값을 cv.GC_PR_BGD(아마 배경)로 설정합니다
mask = np.zeros(image.shape[:2], np.uint8)

# 배경 모델과 전경 모델 초기화
# GrabCut 알고리즘이 사용할 내부 모델입니다
# (1, 65): 가우시안 혼합 모델의 계수 저장
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# ===== ROI (관심 영역) 설정 =====
# 객체를 포함하는 초기 사각형 영역을 설정합니다
# (x, y, width, height) 형식으로 지정
# 커피잔 전체를 포함하도록 여유있게 설정합니다
h, w = image.shape[:2]
x = int(w * 0.05)
y = int(h * 0.05)
width = int(w * 0.90)
height = int(h * 0.90)
rect = (x, y, width, height)

print(f"ROI 정보: x={x}, y={y}, width={width}, height={height}")

# ===== GrabCut 실행 =====
# cv.grabCut(): 이미지에서 객체를 자동으로 분할합니다
# image: 입력 이미지 (BGR)
# mask: 초기 마스크
# bgdModel, fgdModel: 배경/전경 모델
# iterCount=5: 반복 횟수 (높을수록 더 정확하지만 느림)
# mode=cv.GC_INIT_WITH_RECT: 사각형 영역으로 초기화
cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# ===== 마스크 처리 =====
# GrabCut의 결과 마스크는 4가지 값을 가집니다:
# cv.GC_BGD (0): 확실한 배경
# cv.GC_FGD (1): 확실한 전경(객체)
# cv.GC_PR_BGD (2): 아마 배경
# cv.GC_PR_FGD (3): 아마 전경(객체)

# 확실한 전경(1)과 아마 전경(3)을 객체로 선택하고, 나머지는 배경으로 설정합니다
# np.where(): 조건에 따라 값을 선택합니다
# (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD): 전경에 해당하는 픽셀
output_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')

# ===== 배경 제거 =====
# 원본 이미지를 복사합니다
image_no_bg = image.copy()

# 마스크를 3채널로 확장합니다 (BGR에 맞춰서)
mask_3channel = cv.cvtColor(output_mask, cv.COLOR_GRAY2BGR)

# 마스크를 이용하여 배경을 제거합니다
# 마스크 값이 0인 부분(배경)은 검은색(0, 0, 0)으로 만듭니다
image_no_bg = cv.bitwise_and(image_no_bg, mask_3channel)

# ===== 결과 시각화 =====
# 3개의 이미지를 나란히 표시합니다
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 원본 이미지 표시
# 초기 사각형 영역을 표시하기 위해 rectangle을 그립니다
image_with_rect = image.copy()
cv.rectangle(image_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
axes[0].imshow(cv.cvtColor(image_with_rect, cv.COLOR_BGR2RGB))
axes[0].set_title('Original image (Initial region)', fontsize=12, fontweight='bold')
axes[0].axis('off')

# 마스크 이미지 표시
axes[1].imshow(output_mask, cmap='gray')
axes[1].set_title('GrabCut mask (object = white)', fontsize=12, fontweight='bold')
axes[1].axis('off')

# 배경 제거된 이미지 표시
axes[2].imshow(cv.cvtColor(image_no_bg, cv.COLOR_BGR2RGB))
axes[2].set_title('Background removal result (object only)', fontsize=12, fontweight='bold')
axes[2].axis('off')

# 전체 제목 설정
plt.suptitle('GrabCut Object Extraction', fontsize=14, fontweight='bold', y=1.00)

# 레이아웃 자동 조정
plt.tight_layout()

# 결과 이미지 저장 (선택사항)
cv.imwrite('grabcut_mask.jpg', output_mask)
cv.imwrite('grabcut_result.jpg', image_no_bg)
print("마스크가 'grabcut_mask.jpg'로 저장되었습니다.")
print("추출 결과가 'grabcut_result.jpg'로 저장되었습니다.")

# 그래프 표시
plt.show()
