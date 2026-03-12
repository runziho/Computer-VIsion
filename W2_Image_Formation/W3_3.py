import cv2                                     # OpenCV 라이브러리 불러오기
import numpy as np                             # NumPy 라이브러리 불러오기
from pathlib import Path                       # 경로 처리를 위한 Path 불러오기

# 출력 폴더 생성
output_dir = Path("./outputs")                 # 출력 폴더 경로 설정
output_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더가 없으면 생성

# 좌/우 이미지 불러오기
left_color = cv2.imread("images/left.png")            # 왼쪽 이미지 읽기
right_color = cv2.imread("images/right.png")          # 오른쪽 이미지 읽기

if left_color is None or right_color is None:  # 이미지가 정상적으로 읽혔는지 확인
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터
f = 700.0                                      # 초점거리 설정
B = 0.12                                       # 베이스라인 설정

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)     # 왼쪽 이미지를 그레이스케일로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)   # 오른쪽 이미지를 그레이스케일로 변환

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)   # StereoBM 객체 생성
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0   # disparity map 계산

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
valid_mask = disparity > 0                                # disparity가 0보다 큰 유효 픽셀 선택
depth_map = np.zeros_like(disparity, dtype=np.float32)   # depth map 배열 생성
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 유효한 disparity에 대해 depth 계산

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}                                              # ROI별 결과 저장용 딕셔너리 생성

for name, (x, y, w, h) in rois.items():                   # 각 ROI에 대해 반복
    roi_disp = disparity[y:y+h, x:x+w]                    # ROI 영역의 disparity 추출
    roi_depth = depth_map[y:y+h, x:x+w]                   # ROI 영역의 depth 추출
    roi_valid = roi_disp > 0                              # ROI 내부 유효 disparity 마스크 생성

    if np.any(roi_valid):                                 # 유효한 disparity 값이 하나라도 있으면
        mean_disp = np.mean(roi_disp[roi_valid])          # 평균 disparity 계산
        mean_depth = np.mean(roi_depth[roi_valid])        # 평균 depth 계산
    else:                                                 # 유효한 disparity 값이 없으면
        mean_disp = np.nan                                # 평균 disparity를 NaN으로 설정
        mean_depth = np.nan                               # 평균 depth를 NaN으로 설정

    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 계산
# -----------------------------
valid_results = {k: v for k, v in results.items() if not np.isnan(v["mean_depth"])}   # 유효한 depth 값만 저장

if len(valid_results) > 0:                                # 유효한 ROI가 존재하면
    nearest = min(valid_results.items(), key=lambda x: x[1]["mean_depth"])   # 평균 depth가 가장 작은 ROI 찾기
    farthest = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])  # 평균 depth가 가장 큰 ROI 찾기
else:                                                     # 유효한 ROI가 없으면
    nearest = None                                        # nearest를 None으로 설정
    farthest = None                                       # farthest를 None으로 설정

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()                               # disparity 복사본 생성
disp_tmp[disp_tmp <= 0] = np.nan                          # 0 이하 값은 NaN 처리

if np.all(np.isnan(disp_tmp)):                            # 유효한 disparity가 없으면 오류 발생
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)                     # 하위 5퍼센타일 계산
d_max = np.nanpercentile(disp_tmp, 95)                    # 상위 95퍼센타일 계산

if d_max <= d_min:                                        # 최대값이 최소값보다 작거나 같으면
    d_max = d_min + 1e-6                                  # 0으로 나누는 오류 방지

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)        # disparity 정규화
disp_scaled = np.clip(disp_scaled, 0, 1)                  # 0~1 범위로 제한

disp_vis = np.zeros_like(disparity, dtype=np.uint8)       # disparity 시각화용 배열 생성
valid_disp = ~np.isnan(disp_tmp)                          # 유효 disparity 마스크 생성
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)   # 0~255로 변환

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)   # 컬러맵 적용

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)      # depth 시각화용 배열 생성

if np.any(valid_mask):                                    # 유효한 depth 값이 존재하면
    depth_valid = depth_map[valid_mask]                   # 유효 depth 값 추출

    z_min = np.percentile(depth_valid, 5)                 # 하위 5퍼센타일 계산
    z_max = np.percentile(depth_valid, 95)                # 상위 95퍼센타일 계산

    if z_max <= z_min:                                    # 최대값이 최소값보다 작거나 같으면
        z_max = z_min + 1e-6                              # 0으로 나누는 오류 방지

    depth_scaled = (depth_map - z_min) / (z_max - z_min)  # depth 정규화
    depth_scaled = np.clip(depth_scaled, 0, 1)            # 0~1 범위로 제한
    depth_scaled = 1.0 - depth_scaled                     # 가까울수록 큰 값이 되도록 반전
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)   # 0~255로 변환

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)   # 컬러맵 적용

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()                              # 왼쪽 ROI 표시용 이미지 복사
right_vis = right_color.copy()                            # 오른쪽 ROI 표시용 이미지 복사

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# disparity map에도 ROI 표시
disp_vis_roi = disparity_color.copy()                     # disparity ROI 표시용 이미지 복사

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(disp_vis_roi, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(disp_vis_roi, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "original_roi.png"), left_vis)      # ROI가 표시된 원본 이미지 저장
cv2.imwrite(str(output_dir / "disparity_map.png"), disp_vis_roi) # ROI가 표시된 disparity map 저장

# -----------------------------
# 9. 출력
# -----------------------------
for name, values in results.items():   # 각 ROI 평균 disparity / depth 출력
    print(f"{name} -> 평균 disparity: {values['mean_disparity']:.2f}, 평균 depth: {values['mean_depth']:.2f}")

if nearest is not None and farthest is not None:
    print(f"\n평균 depth 기준 가장 가까운 ROI: {nearest[0]}")
    print(f"평균 depth 기준 가장 먼 ROI: {farthest[0]}")
else:
    print("유효한 depth 값을 가진 ROI가 없습니다.")

cv2.imshow("Original", left_vis)        # ROI 표시된 원본 이미지
cv2.imshow("Disparity map", disp_vis_roi) # disparity 시각화
cv2.imshow("Depth map", depth_color)     # depth 시각화

cv2.waitKey(0)
cv2.destroyAllWindows()