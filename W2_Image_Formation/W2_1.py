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