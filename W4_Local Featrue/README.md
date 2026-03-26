# 1. SIFT를 이용한 특징점 검출 및 시각화

- 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(scale-Invariang Feature Transfoam 알고리즘을 사용하여 특징점을 검출하고 이를 시각화

### 전체 코드
```python
import cv2 as cv                                                        
import matplotlib.pyplot as plt                                         

img = cv.imread('mot_color70.jpg')                                        # 이미지 파일 불러오기
if img is None:                                                           # 이미지 로드 실패 시
    raise FileNotFoundError("이미지를 찾을 수 없습니다: mot_color70.jpg") # 오류 

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)                              # 이미지를 BGR → RGB 변환 (matplotlib용)

sift = cv.SIFT_create(                                                    # SIFT 객체 생성
    nfeatures=300,                                                        # 검출할 최대 특징점 수
    nOctaveLayers=3,                                                      # 옥타브당 레이어 수 (기본값 3)
    contrastThreshold=0.04,                                               # 낮을수록 특징점 많아짐 (기본값 0.04)
    edgeThreshold=10,                                                     # 높을수록 엣지 부근 특징점 많아짐 (기본값 10)
    sigma=1.6                                                             # 가우시안 블러 시그마 (기본값 1.6)
)

keypoints, descriptors = sift.detectAndCompute(img_rgb, mask=None)        # 특징점 및 128차원 디스크립터 계산
print(f"검출된 특징점 수: {len(keypoints)}")                              # 검출된 특징점 개수 출력

img_keypoints = cv.drawKeypoints(                                         # 특징점을 이미지에 시각화
    img_rgb,                                                              # 원본 이미지
    keypoints,                                                            # 검출된 특징점 리스트
    outImage=None,                                                        # 출력 이미지 (None이면 새로 생성)
    color=(0, 255, 0),                                                    # 특징점 색상 (초록색)
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS                       # 특징점 위치 + 크기 + 방향 모두 표시
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))                           # 1행 2열 서브플롯 생성
fig.suptitle('SIFT Feature Detection', fontsize=16, fontweight='bold')    # 전체 제목 설정

axes[0].imshow(img_rgb)                                                   # 1번 칸에 원본 이미지 출력
axes[0].set_title('Original Image', fontsize=13)                          # 1번 칸 칸 제목 설정
axes[0].axis('off')                                                       # 1번 칸 축 숨기기

axes[1].imshow(img_keypoints)                                             # 2번 칸에 특징점 이미지 출력
axes[1].set_title(f'SIFT Keypoints  (n = {len(keypoints)})', fontsize=13) # 2번 칸 제목 설정
axes[1].axis('off')                                                       # 2번 칸 축 숨기기

plt.tight_layout()                                                        # 서브플롯 간격 자동 조정
plt.savefig('sift_result.png', dpi=150, bbox_inches='tight')              # 결과 이미지 파일로 저장
plt.show()                                                                # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_result.png")                           # 저장 완료
```

## 1) cv.SIFT_create()를 사용하여 SIFT 객체를 생성

```python
sift = cv.SIFT_create(                                                    # SIFT 객체 생성
    nfeatures=300,                                                        # 검출할 최대 특징점 수
    nOctaveLayers=3,                                                      # 옥타브당 레이어 수 (기본값 3)
    contrastThreshold=0.04,                                               # 낮을수록 특징점 많아짐 (기본값 0.04)
    edgeThreshold=10,                                                     # 높을수록 엣지 부근 특징점 많아짐 (기본값 10)
    sigma=1.6                                                             # 가우시안 블러 시그마 (기본값 1.6)
)
```

## 2) detectAndCompute()를 사용하여 특징점을 검출

``` python
keypoints, descriptors = sift.detectAndCompute(img_rgb, mask=None)        # 특징점 및 128차원 디스크립터 계산
print(f"검출된 특징점 수: {len(keypoints)}")                              # 검출된 특징점 개수 출력
```

## 3) cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화

```python
img_keypoints = cv.drawKeypoints(                                         # 특징점을 이미지에 시각화
    img_rgb,                                                              # 원본 이미지
    keypoints,                                                            # 검출된 특징점 리스트
    outImage=None,                                                        # 출력 이미지 (None이면 새로 생성)
    color=(0, 255, 0),                                                    # 특징점 색상 (초록색)
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS                       # 특징점 위치 + 크기 + 방향 모두 표시
)
```

## 4) matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))                           # 1행 2열 서브플롯 생성
fig.suptitle('SIFT Feature Detection', fontsize=16, fontweight='bold')    # 전체 제목 설정

axes[0].imshow(img_rgb)                                                   # 1번 칸에 원본 이미지 출력
axes[0].set_title('Original Image', fontsize=13)                          # 1번 칸 칸 제목 설정
axes[0].axis('off')                                                       # 1번 칸 축 숨기기

axes[1].imshow(img_keypoints)                                             # 2번 칸에 특징점 이미지 출력
axes[1].set_title(f'SIFT Keypoints  (n = {len(keypoints)})', fontsize=13) # 2번 칸 제목 설정
axes[1].axis('off')                                                       # 2번 칸 축 숨기기

plt.tight_layout()                                                        # 서브플롯 간격 자동 조정
plt.savefig('sift_result.png', dpi=150, bbox_inches='tight')              # 결과 이미지 파일로 저장
plt.show()                                                                # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_result.png")                           # 저장 완료 메시지 출력
```

### 실행 결과

<img width="2091" height="982" alt="image" src="https://github.com/user-attachments/assets/bc902db3-6653-4883-9932-41c860bd9b6c" />


# 2. SIFT를 이용한 두 영상 간 특징점 매칭
- 두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화

### 전체 코드
```python
import cv2 as cv                                                              
import matplotlib.pyplot as plt                                              

img1 = cv.imread('mot_color70.jpg')                                           # 1번째,
img2 = cv.imread('mot_color83.jpg')                                           # 2번째 이미지 파일 로드
if img1 is None or img2 is None:                                              # 이미지 로드 실패 시
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")                # 오류

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)                                # 이미지 BGR → RGB 변환
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)                                # 이미지 BGR → RGB 변환

sift = cv.SIFT_create(nfeatures=300)                                          # SIFT 객체 생성 (최대 특징점 300개)

kp1, des1 = sift.detectAndCompute(img1_rgb, mask=None)                        # 첫 번째 이미지 특징점 및 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_rgb, mask=None)                        # 두 번째 이미지 특징점 및 디스크립터 계산
print(f"img1 특징점 수: {len(kp1)},  img2 특징점 수: {len(kp2)}")            # 각 이미지 특징점 개수 출력

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)                                # BFMatcher 생성 (L2 거리, 교차 검사 활성화)
matches = bf.match(des1, des2)                                                # 두 디스크립터 간 1:1 매칭 수행
matches = sorted(matches, key=lambda x: x.distance)                          # 거리 기준 오름차순 정렬 (좋은 매칭 우선)
print(f"매칭 수: {len(matches)}")                                             # 매칭 개수 출력

img_matches = cv.drawMatches(                                                 # 매칭 결과 이미지 생성
    img1_rgb, kp1,                                                            # 첫 번째 이미지와 특징점
    img2_rgb, kp2,                                                            # 두 번째 이미지와 특징점
    matches[:50],                                                             # 상위 50개 매칭만 표시
    outImg=None,                                                              # 출력 이미지 (None이면 새로 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS                          # 매칭되지 않은 특징점은 표시 안 함
)

plt.figure(figsize=(14, 6))                                                   # 그림 크기 설정
plt.imshow(img_matches)                                                       # 매칭 결과 이미지 출력
plt.title(f'BFMatcher Result (Top 50 / {len(matches)})', fontsize=13)        # 제목 설정
plt.axis('off')                                                               # 축 숨기기
plt.tight_layout()                                                            # 여백 자동 조정
plt.savefig('sift_matching_result.png', dpi=150, bbox_inches='tight')         # 결과 이미지 파일로 저장
plt.show()                                                                    # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_matching_result.png")                      # 저장 완료 메시지 출력
```

## 1) cv.imread()를 사용하여 두 개의 이미지를 불러옴
```python
img1 = cv.imread('mot_color70.jpg')                                           # 1번째,
img2 = cv.imread('mot_color80.jpg')                                           # 2번째 이미지 파일 로드
if img1 is None or img2 is None:                                              # 이미지 로드 실패 시
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")                # 오류
```
## 2) cv.SIFT_create()를 사용하여 특징점을 추출
```python
sift = cv.SIFT_create(nfeatures=300)                                          # SIFT 객체 생성 (최대 특징점 300개)

kp1, des1 = sift.detectAndCompute(img1_rgb, mask=None)                        # 첫 번째 이미지 특징점 및 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_rgb, mask=None)                        # 두 번째 이미지 특징점 및 디스크립터 계산
print(f"img1 특징점 수: {len(kp1)},  img2 특징점 수: {len(kp2)}") 
```
## 3) cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭
```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)                                # BFMatcher 생성 (L2 거리, 교차 검사 활성화)
matches = bf.match(des1, des2)                                                # 두 디스크립터 간 1:1 매칭 수행
matches = sorted(matches, key=lambda x: x.distance)                          # 거리 기준 오름차순 정렬 (좋은 매칭 우선)
print(f"매칭 수: {len(matches)}")          
```
## 4) cv.drawMatches()를 사용하여 매칭 결과를 시각화
```python
img_matches = cv.drawMatches(                                                 # 매칭 결과 이미지 생성
    img1_rgb, kp1,                                                            # 첫 번째 이미지와 특징점
    img2_rgb, kp2,                                                            # 두 번째 이미지와 특징점
    matches[:50],                                                             # 상위 50개 매칭만 표시
    outImg=None,                                                              # 출력 이미지 (None이면 새로 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS                          # 매칭되지 않은 특징점은 표시 안 함
)
```

## 5) matplotlib을 이용하여 매칭 결과를 출력
```python
plt.figure(figsize=(14, 6))                                                   # 그림 크기 설정
plt.imshow(img_matches)                                                       # 매칭 결과 이미지 출력
plt.title(f'BFMatcher Result (Top 50 / {len(matches)})', fontsize=13)        # 제목 설정
plt.axis('off')                                                               # 축 숨기기
plt.tight_layout()                                                            # 여백 자동 조정
plt.savefig('sift_matching_result.png', dpi=150, bbox_inches='tight')         # 결과 이미지 파일로 저장
plt.show()                                                                    # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_matching_result.png")        
```

## 실행 결과


# 3. 호모그래피를 이용한 이미지 정합 (Image Alignment)
- SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬
- 샘플파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택

## 전체 코드
```python
import cv2 as cv             
import numpy as np
import matplotlib.pyplot as plt 

img = cv.imread('coffee cup.JPG')  # 이미지 파일 불러오기
img_copy = img.copy()  # 원본용 복사

rect = cv.selectROI("Select Object", img, False)  # (x, y, width, height) 형태로 자동 생성됨
cv.destroyAllWindows()  # ROI 선택 창 닫기

mask = np.zeros(img.shape[:2], np.uint8)  # 이미지 크기와 동일한 마스크 생성 (초기값 0)
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델 초기화
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델 초기화

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)  # 사용자가 지정한 영역(rect)을 기준으로 GrabCut 수행 (배경/객체 분리)

mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')  # 배경(0), 객체(1)로 마스크 재구성

result = img * mask2[:, :, np.newaxis]  # 마스크를 이용해 객체 부분만 남기고 배경 제거

plt.figure(figsize=(15,5))  # 전체 출력 화면 크기 설정

plt.subplot(1,3,1)  # 1번째 이미지
plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))  # 원본 이미지 출력 (BGR → RGB 변환)
plt.title('Original Image')  # 제목 설정
plt.axis('off')  # 축 제거

plt.subplot(1,3,2)  # 2번째 이미지
plt.imshow(mask2, cmap='gray')  # 마스크 이미지 출력 (흑백)
plt.title('Mask')  # 제목 설정
plt.axis('off')  # 축 제거

plt.subplot(1,3,3)  # 3번 이미지
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))  # 배경 제거된 결과 이미지 출력
plt.title('Result (Object Only)')  # 제목 설정
plt.axis('off')  # 축 제거

plt.show()  # 모든 결과 화면에 출력
```
## 1) cv.imread()를 사용하여 두 개의 이미지를 불러옴
```python
mask = np.zeros(img.shape[:2], np.uint8)  # 이미지 크기와 동일한 마스크 생성 (초기값 0)
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델 초기화
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델 초기화

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)  # 사용자가 지정한 영역(rect)을 기준으로 GrabCut 수행 (배경/객체 분리)
```

## 2) Cv.SIFT_create()를 사용하여 특징점을 검출
```python
rect = cv.selectROI("Select Object", img, False)  # (x, y, width, height) 형태로 자동 생성
```

## 3) cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
```python
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')  # 배경(0), 객체(1)로 마스크 재구성

result = img * mask2[:, :, np.newaxis]  # 마스크를 이용해 객체 부분만 남기고 배경 제거
```

## 4) cv.findHomography()를 사용하여 호모그래피 행렬을 계산

```python
plt.subplot(1,3,1)  # 1번째 이미지
plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))  # 원본 이미지 출력 (BGR → RGB 변환)
plt.title('Original Image')  # 제목 설정
plt.axis('off')  # 축 제거

plt.subplot(1,3,2)  # 2번째 이미지
plt.imshow(mask2, cmap='gray')  # 마스크 이미지 출력 (흑백)
plt.title('Mask')  # 제목 설정
plt.axis('off')  # 축 제거

plt.subplot(1,3,3)  # 3번 이미지
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))  # 배경 제거된 결과 이미지 출력
plt.title('Result (Object Only)')  # 제목 설정
plt.axis('off')  # 축 제거

plt.show()  # 모든 결과 화면에 출력
```

## 5) cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
```python

```

## 6) 변환된 이미지(Warperd Image)와 특징점 매칭 결과(Macthing Result)를 나란히 출력
```python

```
## 출력 결과
- 실행시 화면
