# 1. 체크보드 기반 카메라 캘리브레이션

- 모든 이미지에서 체크보드 코너를 검출
- 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성
- cv2.calibrateCamera()를 사용ㅇ하여 카메라 내부 행렬 k와 왜곡 계수를 구함
- cv2.undistoer()를 사용하여 왜곡 보정한 결과를 시각화

### 개념

### 전체 코드
```python
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
```

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

### 전체 코드
```python
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
```

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

### 전체 코드

```python
import cv2 as cv
import sys

img = cv.imread('girl_laughing.jpg') #img 로드

if img is None : 
    sys.exit('파일이 존재하지 않습니다.') #img파일이 없다면 종료

imgs = img.copy() #원본용 이미지 저장해두기

ix, iy = None, None #시작 좌표 
drawing = False #드래그 중 여부
roi = None #선택된 영역

def draw(event,x,y,flags,param):#마우스 이벤트 함수
    global ix,iy,drawing,img,roi #전역변수

    if event == cv.EVENT_LBUTTONDOWN: #마우스 왼쪽 버튼 클릭했을 떄 초기 위치 저장
        drawing = True #드래그 상태 
        ix,iy = x,y #현재 위치를 시작 좌표로 저장
    
    elif event == cv.EVENT_MOUSEMOVE: #드래그 중이면 사각형 표시
        if drawing:
            img = imgs.copy() #계속 새로 그림
            cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2) #사각형 그림
    
    elif event == cv.EVENT_LBUTTONUP: #마우스 떼면 ROI 선택된것 별도의 창에 출력
        drawing = False
        img = imgs.copy() #계속 새로 그림
        cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2) #사각형 그리기

        #numpy 슬라이싱으로 roi 추출
        x1,x2 = min(ix,x), max(ix,x)
        y1,y2 = min(iy,y), max(iy,y)

        roi = imgs[y1:y2, x1:x2] #numpy 슬라이싱을 사용하여 roi 저장

        if roi.size !=0:
            cv.imshow('ROI',roi) #roi가 존재하면 (0이아니면) ROI창 띄움

    cv.imshow('Drawing',img)

cv.namedWindow('Drawing') #윈도우 이름 Drawing로 선언  
cv.setMouseCallback('Drawing', draw) #마우스 이벤트 처리

while True: #계속 반복
    cv.imshow('Drawing', img) #Drawing 창 만들기

    key = cv.waitKey(1) & 0xFF   #키 입력 받기

    if key == ord('r'): # r 입력받을 시 선택 리셋
        img = imgs.copy()
        cv.destroyWindow('ROI')  # ROI 창 닫기

    elif key == ord('s'): #s입력받으면 ROI img 저장
        if roi is not None:
            cv.imwrite('roi.jpg', roi) #ROI 저장
            print('ROI saved') #ROI 저장 완료 출력

    elif key == ord('q'): # q 입력시 종료
        break

cv.destroyAllWindows() #Open cv가 만든 창을 완전히 닫음
```


## 1) 이미지를 불러오고 화면에 출력

```python
img = cv.imread('girl_laughing.jpg') #img 로드

...
    cv.imshow('Drawing',img)

```

## 2) cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리

```python
cv.setMouseCallback('Drawing', draw) #draw 함수에 대한 마우스 이벤트 처리
```

## 3) 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택

```python
    if event == cv.EVENT_LBUTTONDOWN: #마우스 왼쪽 버튼 클릭했을 떄 초기 위치 저장
        drawing = True #드래그 상태 
        ix,iy = x,y #현재 위치를 시작 좌표로 저장
    
    elif event == cv.EVENT_MOUSEMOVE: #드래그 중이면 사각형 표시
        if drawing:
            img = imgs.copy() #계속 새로 그림(imgs는 img 리셋용으로 저장해둔 것)
            cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2) #사각형 그림
```
출력 결과 
<img width="1400" height="800" alt="image" src="https://github.com/user-attachments/assets/5a460bd2-b509-4263-b21a-840e9fa224d1" />



## 4) 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력

```python    
    elif event == cv.EVENT_LBUTTONUP: #마우스 떼면 ROI 선택된것 별도의 창에 출력
        drawing = False
        img = imgs.copy() #계속 새로 그림
        cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2) #사각형 그리기

        #numpy 슬라이싱으로 roi 추출
        x1,x2 = min(ix,x), max(ix,x)
        y1,y2 = min(iy,y), max(iy,y)

        roi = imgs[y1:y2, x1:x2] #numpy 슬라이싱을 사용하여 roi 저장

        if roi.size !=0:
            cv.imshow('ROI',roi) #roi가 존재하면 (0이아니면) ROI창 띄움

    cv.imshow('Drawing',img)
```
출력 결과
<img width="1400" height="800" alt="image" src="https://github.com/user-attachments/assets/3b902002-0406-4aeb-8740-87e5f366df16" />



## 5) r키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택, s키를 누르면 선택한 영역을 이미지 파일로 저장

```python
    if key == ord('r'): # r 입력받을 시 선택 리셋
        img = imgs.copy()
        cv.destroyWindow('ROI')  # ROI 창 닫기

    elif key == ord('s'): #s입력받으면 ROI img 저장
        if roi is not None:
            cv.imwrite('roi.jpg', roi) #ROI 저장
            print('ROI saved') #ROI 저장 완료 출력
```


### 출력 결과
<img width="2032" height="604" alt="image" src="https://github.com/user-attachments/assets/81834d9f-5d1d-4e6c-a6d2-9a2412312be6" />
















