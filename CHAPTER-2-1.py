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

