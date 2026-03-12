# 1. 체크보드 기반 카메라 캘리브레이션

- 모든 이미지에서 체크보드 코너를 검출
- 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성
- cv2.calibrateCamera()를 사용ㅇ하여 카메라 내부 행렬 k와 왜곡 계수를 구함
- cv2.undistoer()를 사용하여 왜곡 보정한 결과를 시각화


<details>
    <summary>전체 코드</summary>
                
        import cv2
        import numpy as np
        import glob
        import os
        
        # 체크보드 내부 코너 개수
        CHECKERBOARD = (9, 6)
        
        # 체크보드 한 칸 실제 크기 (mm)
        square_size = 25.0
        
        # 코너 정밀화 조건
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 실제 좌표 생성
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) # 체크보드의 실제 3D 좌표 배열을 생성
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # 체크보드의 평면 좌표를 생성
        objp *= square_size # 실제 칸 크기를 반영하여 좌표를 mm 단위로 변환
        
        # 저장할 좌표
        objpoints = [] # 각각의 이미지 실제 3D 좌표를 저장할 리스트를 생성
        imgpoints = [] # 각각의 이미지 2D 좌표를 저장할 리스트를 생성
        
        # 첫 번째 성공 이미지 저장용
        valid_image_path = None
        
        # 이미지 불러오기
        images = sorted(glob.glob("images/calibration_images/left*.jpg"))
        
        print("찾은 이미지 개수:", len(images)) # 불러온 이미지 개수를 출력
        
        img_size = None # 이미지 크기를 저장할 변수를 초기화
        
        # -----------------------------
        # 1. 체크보드 코너 검출
        # -----------------------------
        for fname in images: # 모든 캘리브레이션 이미지에 대해 반복
            img = cv2.imread(fname)  # 현재 이미지 읽어옴
        
            if img is None: # 이미지 읽지 못하면
                continue # 건너뜀
        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #컬러 이미지를 그레이스케일 이미지로 변환
            img_size = gray.shape[::-1] # 이미지 크기를 width, height 형태로 저장
        
            ret, corners = cv2.findChessboardCorners( 
                gray,
                CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE 
            ) #체크보드 이미지에서 내부 코너의 2D 좌표 검출
        
            if ret: # 코너 검출에 성공한 경우
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) #검출된 코너 위치 정밀하게 보정
        
                objpoints.append(objp.copy()) #현재 이미지에 대응하는 실제 3D 좌표 저장
                imgpoints.append(corners2) #현재 이미지를 왜곡 보정용 예시 이미지로 저장
        
                if valid_image_path is None: # 첫 번쨰 성공 이미지가 아직 저장되지 않은 경우
                    valid_image_path = fname #현재 이미지를 왜곡 보정용 예시 이미지로 저장
        
        print("코너 검출 성공 이미지 수:", len(objpoints)) #코너 검출에 성공한 이미지 개수를 출력
        
        if len(objpoints) == 0: #코너 검출 성공 이미지가 없는 경우
            raise ValueError("체크보드 코너가 검출된 이미지가 없습니다.") #캘리브레이션 불가 메세지 출력
        
        # -----------------------------
        # 2. 카메라 캘리브레이션
        # -----------------------------
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            img_size,
            None,
            None
        ) #실제 3D 좌표와 이미지 2D 좌표를 이용해 카메라 파라미터 계산
        
        print("\nCamera Matrix K:") #카메라 행렬 제목 출력
        print(K) #계산된 내부 파라미터 행렬 출력
        
        print("\nDistortion Coefficients:") #왜곡 계수 제목 출력
        print(dist) #계산된 왜곡 계수 출력
        
        # -----------------------------
        # 3. 왜곡 보정 시각화
        # -----------------------------
        test_img = cv2.imread(valid_image_path) #왜곡 보정에 사용할 이미지 불러옴
        
        if test_img is None: # 왜곡 보정용 이미지 실패 시
            raise ValueError("왜곡 보정용 이미지를 읽을 수 없습니다.")
        
        undistorted = cv2.undistort(test_img, K, dist, None, K) #계산된 파라미터를 이용하여 이미지 왜곡 보정
        
        result = np.hstack((test_img, undistorted)) #원본 이미지와 보정 이미지를 가로로 이어 붙임
        
        cv2.imshow("Original (Left) | Undistorted (Right)", result) #원본 이미지와 보정 이미지 같이 출력
        cv2.imwrite("undistorted_result.jpg", result) #비교 결과 이미지 파일로 저장
        cv2.waitKey(0) # 키 입력 기다림
        cv2.destroyAllWindows() #모든 출력 창 종료
</details>



## 1) 모든 이미지에서 체크보드 코너를 검출

```python
    ret, corners = cv2.findChessboardCorners( 
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE 
    ) #체크보드 이미지에서 내부 코너의 2D 좌표 검출
```

## 2) 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성

``` python
    if ret: # 코너 검출에 성공한 경우
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) #검출된 코너 위치 정밀하게 보정

        objpoints.append(objp.copy()) #현재 이미지에 대응하는 실제 3D 좌표 저장
        imgpoints.append(corners2) #현재 이미지를 왜곡 보정용 예시 이미지로 저장

        if valid_image_path is None: # 첫 번쨰 성공 이미지가 아직 저장되지 않은 경우
            valid_image_path = fname #현재 이미지를 왜곡 보정용 예시 이미지로 저장
```

## 3) cv2.calibrateCamera()를 사용하여 카메라 내부 행렬 k와 왜곡 계수를 구함

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
) #실제 3D 좌표와 이미지 2D 좌표를 이용해 카메라 파라미터 계산
```

## 4) cv2.undistoer()를 사용하여 왜곡 보정한 결과를 시각화

```python
undistorted = cv2.undistort(test_img, K, dist, None, K) #계산된 파라미터를 이용하여 이미지 왜곡 보정
```

### 실행 결과 

<img width="1917" height="755" alt="image" src="https://github.com/user-attachments/assets/9ac14946-307d-4aca-b5e0-39c10c553523" />

<img width="707" height="183" alt="image" src="https://github.com/user-attachments/assets/5bb0da88-d0c0-4314-8758-57af34bc2848" />




# 2. 이미지 Rotation & Transformation
- 한 장의 이미지에 회전, 크기 조절, 평행이동을 적용

<details>
    <summary>전체 코드</summary>
                

    import cv2
    import numpy as np
    
    img = cv2.imread("images/rose.png")   # 이미지 파일 읽기
    
    h, w = img.shape[:2]       # 이미지 높이와 너비 저장
    
    center = (w / 2, h / 2)    # 이미지 중심 좌표 계산
    
    M = cv2.getRotationMatrix2D(center, 30, 0.8)   # 30도 회전 + 0.8배 축소
    
    M[0,2] += 80               # x 방향 +80 이동
    M[1,2] += -40              # y 방향 -40 이동
    
    result = cv2.warpAffine(img, M, (w, h))   # 원본 크기 그대로 변환
    
    combined = np.hstack((img, result))       # 원본과 변환 이미지 가로로 붙이기
    
    scale = 0.7                               # 화면에 맞게 약간 축소
    display = cv2.resize(combined, None, fx=scale, fy=scale)   # 원본과 변환 이미지를 합친 combined 이미지를 scale 비율만큼 축소하여 화면에 맞게 조절
    cv2.imshow("Original | Transformed", display) #합친 이미지 출력
    cv2.waitKey(0) # 키 입력 기다림
    cv2.destroyAllWindows() #창 닫기

</details>

## 1) 이미지의 중심 기준으로 +30도 회전
## 2) 회전과 동시에 크기를 0.8로 조절

```python
center = (w / 2, h / 2)    # 이미지 중심 좌표 계산, w는 너비 h는 높이

M = cv2.getRotationMatrix2D(center, 30, 0.8)   # 중심 기준으로 30도 회전하고 크기를 0.8배로 조절하는 행렬 생성
```

## 3) 그 결과를 x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동
```python
M[0, 2] += 80              # x축 방향으로 80픽셀 이동
M[1, 2] += -40             # y축 방향으로 -40픽셀 이동
```

### 실행 결과

<img width="2490" height="865" alt="image" src="https://github.com/user-attachments/assets/2d777fcf-ccde-49e5-9911-dd8b28f5825c" />




# 3. Stereo Disparity 기반 Depth 추정

- 같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용해 깊이를 추정
- 두 이미지에서 같은 물체가 얼마나 옆으로 이동해 보이는지 계산하여 물체가 카메라에서 얼마나 떨어져있는지(depth)를 구할 수 있음

<details>
    <summary>전체 코드</summary>
    
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
    cv2.imshow("Depth map", depth_color)     # depth 시각화 빨강 -> 가까움, 파랑 -> 멀리 있음
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
</details>

## 1) 입력 이미지를 그레이스케일로 변환한 뒤 cv2.StereoBM_create()를 사용하여 disparity map 계산

```python
# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)     # 왼쪽 이미지를 그레이스케일로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)   # 오른쪽 이미지를 그레이스케일로 변환

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)   # StereoBM 객체 생성
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0   # disparity map 계산

```

## 2) DIsparity > 0인 픽셀만 사용하여 depth map 계산

```python
# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
valid_mask = disparity > 0                                # disparity가 0보다 큰 유효 픽셀 선택
depth_map = np.zeros_like(disparity, dtype=np.float32)   # depth map 배열 생성
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 유효한 disparity에 대해 depth 계산

```

## 3) ROI PAinting, Frog, Teddy 각각에 대해 평균 disparity와 평균 depth를 계산

```python
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

```


## 4) 세 ROI중 어떤 영역이 가장 가까운지, 어떤 영역이 가장 먼지 해석

```python    
valid_results = {k: v for k, v in results.items() if not np.isnan(v["mean_depth"])}   # 유효한 depth 값만 저장

if len(valid_results) > 0:                                # 유효한 ROI가 존재하면
    nearest = min(valid_results.items(), key=lambda x: x[1]["mean_depth"])   # 평균 depth가 가장 작은 ROI 찾기
    farthest = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])  # 평균 depth가 가장 큰 ROI 찾기
else:
    nearest = None
    farthest = None
```

출력 부분
```python
if nearest is not None and farthest is not None:
    print(f"\n평균 depth 기준 가장 가까운 ROI: {nearest[0]}")
    print(f"평균 depth 기준 가장 먼 ROI: {farthest[0]}")
else:
    print("유효한 depth 값을 가진 ROI가 없습니다.")
```



### 출력 결과
<img width="2032" height="604" alt="image" src="https://github.com/user-attachments/assets/81834d9f-5d1d-4e6c-a6d2-9a2412312be6" />

<img width="692" height="209" alt="image" src="https://github.com/user-attachments/assets/1f6b6373-c1b4-4d42-b186-9b053adcb31d" />













