# chapter 2
## 과제1 설명 및 요구사항 (체크보드 기반 카메라 캘리브레이션)
 - 이미지에서 체크보드 코너를 검출하고 실제 좌표와 이미지 좌표의 대응 관계를 이용하여 카메라 파라미터 추정
 - 체크보드 패턴이 촬영된 여러 장의 이미지를 이용하여 카메라의 내부 행렬과 왜곡 계수를 계산하여 왜곡 보정
 - 모든 이미지에서 체크보드 코너를 검출
 - 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성
 - cv2.calibrateCamera()를 사용하여 카메라 내부 행렬k와 왜곡계수를 구함
 - cv2.undistort()를 사용하여 왜곡 보정한 결과를 시각화

한줄 요약 - 
<img width="1347" height="649" alt="1" src="https://github.com/user-attachments/assets/fc9c8a25-4451-4a3d-a236-28a405cd0093" />


과제 1번 전체 코드

```python
import cv2
import numpy as np
import glob

# 이 예제의 목표:
# 여러 장의 체크보드 이미지를 이용해 카메라의 내부 파라미터를 추정하고,
# 추정한 결과를 이용해 렌즈 왜곡이 제거된 영상을 확인한다.

# 체크보드 내부 코너 개수
# (9, 6)은 검은 칸/흰 칸 개수가 아니라 "내부 코너"의 가로/세로 개수이다.
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 길이(mm)
# 실제 단위를 넣어 주면, 결과 파라미터의 스케일 해석이 가능해진다.
square_size = 25.0

# 코너를 서브픽셀 단위로 정밀화할 때 사용할 반복 종료 조건
# 30번 반복하거나, 이동량이 0.001 이하로 충분히 작아지면 종료한다.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체크보드의 실제 좌표계 생성
# z=0인 평면 위에 체크보드가 놓여 있다고 가정하고,
# 각 코너의 3차원 좌표를 (x, y, 0) 형태로 만든다.
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 두 리스트는 같은 인덱스끼리 서로 대응된다.
objpoints = [] #각 이미지에서 공통으로 사용하는 실제 세계 좌표들
imgpoints = [] #각 이미지에서 실제로 검출된 2차원 코너 좌표들

images = glob.glob("calibration_images/left*.jpg")

# 모든 입력 이미지가 동일한 해상도인지 확인하기 위한 변수
# calibrateCamera는 이미지 크기 정보가 필요하다.
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
	img = cv2.imread(fname)
	if img is None:
		print(f"[경고] 이미지를 읽을 수 없습니다: {fname}")
		continue

	# 첫 번째 정상 이미지의 크기를 저장한다.
	# 이후 calibrateCamera에 (width, height) 형태로 전달된다.
	if img_size is None:
		img_size = (img.shape[1], img.shape[0])

	# 체크보드 코너 검출은 밝기 정보만으로 충분하므로 그레이스케일로 변환한다.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# 체크보드 내부 코너 탐지
	# ret=True이면 지정한 크기(9x6)의 내부 코너가 모두 검출되었다는 뜻이다.
	# corners에는 각 코너의 픽셀 좌표가 순서대로 저장된다.
	ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

	if ret:
		# 처음 찾은 코너 위치는 정수 픽셀 수준의 근사값일 수 있다.
		# cornerSubPix는 주변 밝기 분포를 이용해 코너를 더 정확한 위치로 보정한다.
		# 캘리브레이션은 코너 위치 오차에 민감하므로 이 단계가 중요하다.
		corners2 = cv2.cornerSubPix(
			gray,
			corners,
			winSize=(11, 11),
			zeroZone=(-1, -1),
			criteria=criteria,
		)

		# 한 장의 이미지에서 얻은 3D-2D 대응 관계를 저장한다.
		# objp는 현실 세계의 코너 좌표, corners는 이미지 위 코너 좌표이다.
		# 여러 장의 이미지에서 이 대응 관계를 모아야 카메라 파라미터를 안정적으로 추정할 수 있다.
		objpoints.append(objp)
		imgpoints.append(corners)

		print(f"[성공] 코너 검출: {fname}")
	else:
		# 코너를 모두 찾지 못하면 정확한 대응 관계를 만들 수 없으므로
		# 해당 이미지는 캘리브레이션에 사용하지 않는다.
		print(f"[실패] 코너 검출 실패(제외): {fname}")


# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 유효한 이미지가 하나도 없으면 더 진행할 수 없다.
if img_size is None:
	raise RuntimeError("사용 가능한 이미지를 찾지 못했습니다. calibration_images 경로를 확인하세요.")

# 코너 검출 성공 이미지가 0장이면 3D-2D 대응 관계 자체가 없으므로 캘리브레이션 불가
if len(objpoints) == 0:
	raise RuntimeError("체크보드 코너를 검출한 이미지가 없습니다. CHECKERBOARD 설정 또는 입력 이미지를 확인하세요.")

# calibrateCamera의 역할
# - K: 카메라 내부 파라미터 행렬
# - dist: 렌즈 왜곡 계수
# - rvecs, tvecs: 각 이미지에서 체크보드가 카메라에 대해 어떤 자세였는지
# 를 동시에 추정한다.
#
# K는 일반적으로 다음 구조를 가진다.
# [ fx   0  cx ]
# [  0  fy  cy ]
# [  0   0   1 ]
# 여기서 fx, fy는 초점거리(픽셀 단위), cx, cy는 주점(principal point)이다.
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
	objpoints,
	imgpoints,
	img_size,
	None,
	None,
)


# 추정된 내부 파라미터와 왜곡 계수를 출력한다.
# dist는 보통 [k1, k2, p1, p2, k3] 형태이며,
# k 계수는 방사 왜곡, p 계수는 접선 왜곡
print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
for fname in images:
	img = cv2.imread(fname)
	if img is None:
		continue

	# undistort는 추정한 카메라 행렬과 왜곡 계수를 사용해
	# 휘어져 보이던 직선과 가장자리 왜곡을 보정한다.
	undistorted = cv2.undistort(img, K, dist, None, K)

	# 원본과 보정 결과를 좌우로 나란히 붙여서 비교한다.
	# 왼쪽은 원본, 오른쪽은 왜곡이 제거된 결과이다.
	comparison = np.hstack((img, undistorted))
	cv2.imshow("Original (Left) vs Undistorted (Right)", comparison)
	# 0은 사용자가 키를 누를 때까지 무한 대기한다.
	# 따라서 이미지마다 결과를 천천히 확인할 수 있다.
	key = cv2.waitKey(0)

	# ESC(27)를 누르면 남은 이미지 확인을 중단한다.
	if key == 27:
		break

# 모든 OpenCV 창을 닫고 프로그램을 마무리한다.
cv2.destroyAllWindows()

```

## 과제2 설명 및 요구사항 (이미지 Rotation & Transformation)
 - 한장의 이미지에 회전, 크기조절, 평행이동을 적용
 - 이미지의 중심 기준으로 +30도 회전
 - 회전과 동시에 크기를 0.8로 조절
 - 그 결과를 x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동

한줄 요약 - 
<img width="2385" height="836" alt="2" src="https://github.com/user-attachments/assets/a28e326e-0844-4992-a16b-5a0288ba2bb6" />

과제 2번 전체 코드

```python
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

```


## 과제3 설명 및 요구사항 (Stereo Disparity 기반 Depth 추정)
 - 같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용
 - 두 이미지에서 같은 물체가 얼마나 옆으로 이동해 보이는지 계산하여 물체가 카메라에서 얼마나 떨어져 있는지(depth)를 구할 수 있음
 - 입력 이미지를 그레이스케일로 변환한 뒤 cv2.StereoBM_create()를 사용하여 disparity map 계산
 - Disparity > 0인 픽셀만 사용하여 depth map 계산
 - ROI Painting, Frog, Teddy 각각에 대해 평균 disparity와 평균 depth를 계산
 - 세 ROI 중 어떤 영역이 가장 가까운지, 어떤 영역이 가장 먼지 해석

한줄 요약 - 
<img width="1438" height="1155" alt="3" src="https://github.com/user-attachments/assets/dcb5a45e-8342-4ca3-bf1c-3d5b6f6d568c" />


과제 3번 전체 코드

```python
import cv2
import numpy as np
from pathlib import Path

# 결과 이미지를 저장할 폴더를 미리 생성한다.
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 카메라 이미지 불러오기
# left.png와 right.png는 동일한 장면을 서로 다른 위치의 카메라에서 찍은 영상이다.
left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
# f: 초점거리(픽셀 단위)
# B: 두 카메라 중심 사이 거리, 즉 baseline(미터 단위)
# 깊이 계산식 Z = fB/d에 직접 사용된다.
f = 700.0
B = 0.12

# ROI(Region of Interest) 설정
# 각 관심 영역은 (x, y, width, height) 형식으로 지정한다.
# 이후 각 영역 안에서 평균 disparity와 평균 depth를 계산해 물체 간 거리를 비교한다.
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# StereoBM은 보통 그레이스케일 영상에서 동작하므로 색상 이미지를 회색조로 변환한다.
# 색상 정보 대신 밝기 패턴을 비교하여 대응점을 찾는다.
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)


# -----------------------------
# 1. Disparity 계산
# -----------------------------
# StereoBM(Block Matching) 객체 생성
# numDisparities: 탐색할 disparity 범위, 16의 배수여야 한다.
# blockSize: 비교에 사용하는 블록 크기, 홀수여야 한다.
# 블록 기반으로 좌/우 영상의 유사한 패턴을 찾아 픽셀 이동량을 추정한다.
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# StereoBM 결과는 내부적으로 16배 스케일된 정수 형태로 반환된다.
# 따라서 실제 disparity 값으로 해석하려면 16.0으로 나누어야 한다.
# 결과는 픽셀 단위의 disparity map이다.
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0


# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
# disparity가 0 이하인 값은 유효한 대응점을 찾지 못했거나 계산이 불안정한 경우이므로 제외한다.
# 문제 조건도 disparity > 0 인 픽셀만 사용하라고 되어 있으므로 그 마스크를 만든다.
valid_mask = disparity > 0

# depth_map은 disparity와 동일한 크기의 실수형 배열로 만든다.
# 유효하지 않은 위치는 0으로 두고, 유효한 위치만 깊이를 계산한다.
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 깊이 계산식 적용: Z = fB / d
# disparity가 클수록 분모가 커져 depth는 작아지므로 물체가 더 가깝다는 뜻이다.
depth_map[valid_mask] = (f * B) / disparity[valid_mask]


# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # 현재 ROI 영역만 잘라서 disparity와 depth를 각각 확인한다.
    roi_disp = disparity[y:y + h, x:x + w]
    roi_depth = depth_map[y:y + h, x:x + w]

    # ROI 내부에서도 disparity가 0 이하인 값은 제외해야 하므로 별도 마스크를 만든다.
    roi_valid = roi_disp > 0

    if np.any(roi_valid):
        # 유효한 disparity 픽셀만 모아 평균 disparity를 계산한다.
        # 이것이 클수록 해당 ROI는 카메라에 더 가깝다.
        mean_disparity = float(np.mean(roi_disp[roi_valid]))

        # 같은 유효 영역에 대해 평균 depth를 계산한다.
        # 이것이 클수록 해당 ROI는 카메라에서 더 멀다.
        mean_depth = float(np.mean(roi_depth[roi_valid]))
    else:
        # ROI 안에 유효 disparity가 하나도 없으면 평균을 낼 수 없으므로 NaN 처리한다.
        mean_disparity = float("nan")
        mean_depth = float("nan")

    # 이름별 결과를 딕셔너리에 저장해 이후 출력과 비교에 사용한다.
    results[name] = {
        "mean_disparity": mean_disparity,
        "mean_depth": mean_depth,
    }

# -----------------------------
# 4. 결과 출력
# -----------------------------
# 각 ROI의 평균 disparity와 평균 depth를 출력한다.
# 발표 시에는 disparity가 큰 영역이 더 가깝고,
# depth가 큰 영역이 더 멀다는 점을 함께 설명하면 된다.
for name, values in results.items():
    print(f"[{name}]")
    print(f"  Mean Disparity: {values['mean_disparity']:.4f}")
    print(f"  Mean Depth: {values['mean_depth']:.4f} m")

# NaN이 포함된 ROI는 비교 대상에서 제외한다.
valid_results = {
    name: values
    for name, values in results.items()
    if not np.isnan(values["mean_disparity"]) and not np.isnan(values["mean_depth"])
}

if len(valid_results) == 0:
    raise ValueError("세 ROI 모두에서 유효한 disparity를 얻지 못했습니다.")

# 가장 가까운 ROI는 평균 disparity가 가장 큰 영역이다.
# 가장 먼 ROI는 평균 depth가 가장 큰 영역이다.
closest_roi = max(valid_results.items(), key=lambda item: item[1]["mean_disparity"])[0]
farthest_roi = max(valid_results.items(), key=lambda item: item[1]["mean_depth"])[0]

print(f"\nNEAREST ROI: {closest_roi}")
print(f"FARTHEST ROI: {farthest_roi}")


# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
# 시각화 전용 복사본을 만든다.
# disparity가 0 이하인 값은 무효값이므로 NaN으로 바꿔 통계 계산에서 제외한다.
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 극단값(outlier)의 영향을 줄이기 위해 5%~95% 백분위 범위를 사용해 정규화한다.
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

# disparity를 0~1 범위로 정규화한 뒤 0~255로 변환해 컬러맵 입력으로 사용한다.
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# JET 컬러맵을 적용하면 값이 큰 쪽이 빨강 계열, 작은 쪽이 파랑 계열로 표현된다.
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
# depth도 시각화를 위해 8비트 영상으로 변환한다.
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    # depth 역시 백분위 기반으로 정규화해 극단값의 영향을 줄인다.
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    # depth는 값이 클수록 더 멀다.
    # 하지만 이번 시각화는 "가까울수록 빨강"으로 표현하고 싶으므로
    # 정규화 후 1 - value로 반전한다.
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# 최종 depth 컬러맵 생성
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
# 원본 이미지를 복사한 뒤 관심 영역을 사각형으로 표시한다.
# 발표 시 어느 영역을 기준으로 평균을 냈는지 바로 보여주기 위해 필요하다.
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    # 왼쪽 이미지에 ROI 사각형과 이름을 표시
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 오른쪽 이미지에도 같은 ROI를 표시해 대응 위치를 시각적으로 확인한다.
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
# 분석 결과를 파일로 저장하면 발표 자료나 보고서에 바로 활용할 수 있다.
cv2.imwrite(str(output_dir / "left_with_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_with_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)

# -----------------------------
# 9. 출력
# -----------------------------
# 최종 결과를 화면에 띄운다.
# 좌/우 원본 영상의 ROI, disparity map, depth map을 한 번에 확인할 수 있다.
cv2.imshow("Left Image with ROI", left_vis)
cv2.imshow("Right Image with ROI", right_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)

# 키 입력을 기다린 뒤 창을 닫는다.
cv2.waitKey(0)
cv2.destroyAllWindows()

```
