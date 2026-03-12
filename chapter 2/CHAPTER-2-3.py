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
