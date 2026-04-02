# 1. 간단한 이미지 분류기 구현

- 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구

### 전체 코드
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data() # MNIST 데이터셋을 로드

# 데이터를 훈련 세트와 테스트 세트로 분할
print("훈련 이미지 shape:", x_train.shape)   # (60000, 28, 28) # 훈련 세트
print("테스트 이미지 shape:", x_test.shape)   # (10000, 28, 28) # 데이터 세트

# 데이터 전처리 
x_train = x_train / 255.0 # 픽셀값을 0~255 -> 0~1로 정규화
x_test = x_test / 255.0 

y_train = to_categorical(y_train, 10) # One-Hot 인코딩
y_test = to_categorical(y_test, 10)

#간단한 신경망 모델을 구축
model = Sequential([             # Sequential 모델 활용
    Flatten(input_shape=(28, 28)),   # 28x28 이미지를 1차원으로 펼침
    Dense(128, activation='relu'),   # Dense 레이어를 활용 / 은닉층
    Dense(10, activation='softmax')  # 출력층: 숫자 0~9 총 10개
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)

print("테스트 손실:", test_loss)
print("테스트 정확도:", test_acc)
```

## 1) MNIST 데이터셋을 로드

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data() # MNIST 데이터셋을 로드
```

## 2) 데이터를 훈련 세트와 테스트 세트로 분할

``` python
print("훈련 이미지 shape:", x_train.shape)   # (60000, 28, 28) # 훈련 세트
print("테스트 이미지 shape:", x_test.shape)   # (10000, 28, 28) # 데이터 세트                     
```

## 3) 간단한 신경망 모델을 구축

- Sequential 모델과 Dense 레이어 활용
```python
model = Sequential([             # Sequential 모델 활용
    Flatten(input_shape=(28, 28)),   # 28x28 이미지를 1차원으로 펼침
    Dense(128, activation='relu'),   # Dense 레이어를 활용 / 은닉층
    Dense(10, activation='softmax')  # 출력층: 숫자 0~9 총 10개
])
```

## 4) 모델을 훈련시키고 정확도를 평가

```python
# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)

print("테스트 손실:", test_loss)
print("테스트 정확도:", test_acc)
```

### 실행 결과

![실행 결과](result_1.png)


# 2. CIFAR-10 데이터셋을활용한 CNN 모델구축
- CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행

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
![매칭 결과](sift_matching_result.png)

# 3. 호모그래피를 이용한 이미지 정합 (Image Alignment)
- SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬
- 샘플파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택

## 전체 코드
```python
import cv2 as cv                                                                      
import matplotlib.pyplot as plt                                                    
import numpy as np                                                                   

img1 = cv.imread('img1.jpg')   # img 1 로드
img2 = cv.imread('img2.jpg')   # img2 로드
if img1 is None or img2 is None:                                                       # 이미지 로드 실패 시
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")                         # 오류
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)                                         # img 1 BGR → RGB 변환
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)                                         # img 2 BGR → RGB 변환

sift = cv.SIFT_create()                                                                # SIFT 객체 생성

kp1, des1 = sift.detectAndCompute(img1_rgb, mask=None)                                 # img1의 특징점 및 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_rgb, mask=None)                                 # img2의 특징점 및 디스크립터 계산
print(f"img1 특징점 수: {len(kp1)},  img2 특징점 수: {len(kp2)}")                     # 각 이미지 특징점 개수 출력

bf = cv.BFMatcher()                                                                    # BFMatcher 객체 생성 (기본: L2 거리)
knn_matches = bf.knnMatch(des1, des2, k=2)                                             # 각 특징점에 대해 최근접 이웃 2개 매칭

good_matches = []                                                                      # 좋은 매칭 결과를 저장할 리스트
for m, n in knn_matches:                                                               # 매칭 쌍 순회
    if m.distance < 0.7 * n.distance:                                                  # Lowe's ratio test: 거리 비율이 0.7 미만이면
        good_matches.append(m)                                                         # 좋은 매칭으로 채택
print(f"좋은 매칭 수: {len(good_matches)}")                                            # 좋은 매칭 개수 출력

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])                         # img1의 매칭 특징점 좌표 추출
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])                         # img2의 매칭 특징점 좌표 추출

H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)                               # RANSAC으로 호모그래피 행렬 계산 (이상점 제거)
print(f"호모그래피 행렬:\n{H}")                                                        # 계산된 호모그래피 행렬 출력

h1, w1 = img1_rgb.shape[:2]                                                            # img1 높이·너비 추출
h2, w2 = img2_rgb.shape[:2]                                                            # img2 높이·너비 추출
panorama_w = w1 + w2                                                                   # 파노라마 너비: 두 이미지 너비 합산
panorama_h = max(h1, h2)                                                               # 파노라마 높이: 두 이미지 중 큰 높이 사용

warped = cv.warpPerspective(img2_rgb, H, (panorama_w, panorama_h))                    # img2를 호모그래피로 변환하여 img1 시점에 정렬
warped[0:h1, 0:w1] = img1_rgb                                                          # 변환된 이미지 위에 img1을 덮어씌워 합성

match_draw = cv.drawMatches(                                                           # 매칭 결과 이미지 생성
    img1_rgb, kp1,                                                                     # img2와 특징점
    img2_rgb, kp2,                                                                     # img2와 특징점
    good_matches[:50],                                                                 # 상위 50개 좋은 매칭만 표시
    outImg=None,                                                                       # 출력 이미지 (None이면 새로 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS                                   # 매칭되지 않은 특징점은 표시 안 함
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))                                        # 1행 2열 서브플롯 생성
fig.suptitle('SIFT Homography', fontsize=16, fontweight='bold')                        # 전체 제목 설정

axes[0].imshow(match_draw)                                                             # 1번 칸에 매칭 결과 출력
axes[0].set_title(f'Matching Result ({len(good_matches)} matches)', fontsize=13)       # 1번째 칸 제목 설정
axes[0].axis('off')                                                                    # 1번째 칸 축 숨기기

axes[1].imshow(warped)                                                                 # 2번 칸에 변환·합성된 이미지 출력
axes[1].set_title('Warped Image (Panorama)', fontsize=13)                              # 2번째 칸 제목 설정
axes[1].axis('off')                                                                    # 2번째 칸 축 숨기기

plt.tight_layout()                                                                     # 서브플롯 간격 자동 조정
plt.savefig('sift_homography_result.png', dpi=150, bbox_inches='tight')                # 결과 이미지 파일로 저장
plt.show()                                                                             # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_homography_result.png")                             # 저장 완료 메시지 출력
```

## 1) cv.imread()를 사용하여 두 개의 이미지를 불러옴
```python
img1 = cv.imread('img1.jpg')   # img 1 로드
img2 = cv.imread('img2.jpg')   # img2 로드
if img1 is None or img2 is None:                                                       # 이미지 로드 실패 시
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")                         # 오류
```

## 2) Cv.SIFT_create()를 사용하여 특징점을 검출
```python
sift = cv.SIFT_create()                                                                # SIFT 객체 생성

kp1, des1 = sift.detectAndCompute(img1_rgb, mask=None)                                 # img1의 특징점 및 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_rgb, mask=None)                                 # img2의 특징점 및 디스크립터 계산
print(f"img1 특징점 수: {len(kp1)},  img2 특징점 수: {len(kp2)}")                     # 각 이미지 특징점 개수 출력
```

## 3) cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
```python
bf = cv.BFMatcher()                                                                    # BFMatcher 객체 생성 (기본: L2 거리)
knn_matches = bf.knnMatch(des1, des2, k=2)                                             # 각 특징점에 대해 최근접 이웃 2개 매칭

good_matches = []                                                                      # 좋은 매칭 결과를 저장할 리스트
for m, n in knn_matches:                                                               # 매칭 쌍 순회
    if m.distance < 0.7 * n.distance:                                                  # Lowe's ratio test: 거리 비율이 0.7 미만이면
        good_matches.append(m)                                                         # 좋은 매칭으로 채택
print(f"좋은 매칭 수: {len(good_matches)}")                                            # 좋은 매칭 개수 출력
```

## 4) cv.findHomography()를 사용하여 호모그래피 행렬을 계산

```python
H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)                               # RANSAC으로 호모그래피 행렬 계산 (이상점 제거)
print(f"호모그래피 행렬:\n{H}")                                                        # 계산된 호모그래피 행렬 출력
```

## 5) cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
```python
h1, w1 = img1_rgb.shape[:2]                                                            # img1 높이·너비 추출
h2, w2 = img2_rgb.shape[:2]                                                            # img2 높이·너비 추출
panorama_w = w1 + w2                                                                   # 파노라마 너비: 두 이미지 너비 합산
panorama_h = max(h1, h2)                                                               # 파노라마 높이: 두 이미지 중 큰 높이 사용

warped = cv.warpPerspective(img2_rgb, H, (panorama_w, panorama_h))                    # img2를 호모그래피로 변환하여 img1 시점에 정렬
warped[0:h1, 0:w1] = img1_rgb                                                          # 변환된 이미지 위에 img1을 덮어씌워 합성
```

## 6) 변환된 이미지(Warperd Image)와 특징점 매칭 결과(Macthing Result)를 나란히 출력
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))                                        # 1행 2열 서브플롯 생성
fig.suptitle('SIFT Homography', fontsize=16, fontweight='bold')                        # 전체 제목 설정

axes[0].imshow(match_draw)                                                             # 1번 칸에 매칭 결과 출력
axes[0].set_title(f'Matching Result ({len(good_matches)} matches)', fontsize=13)       # 1번째 칸 제목 설정
axes[0].axis('off')                                                                    # 1번째 칸 축 숨기기

axes[1].imshow(warped)                                                                 # 2번 칸에 변환·합성된 이미지 출력
axes[1].set_title('Warped Image (Panorama)', fontsize=13)                              # 2번째 칸 제목 설정
axes[1].axis('off')                                                                    # 2번째 칸 축 숨기기

plt.tight_layout()                                                                     # 서브플롯 간격 자동 조정
plt.savefig('sift_homography_result.png', dpi=150, bbox_inches='tight')                # 결과 이미지 파일로 저장
plt.show()                                                                             # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_homography_result.png")                             # 저장 완료 메시지 출력
```
## 출력 결과
- 실행 결과
![실행 결과](sift_homography_result.png)

- 호모그래피 행렬
![실행 결과](matrix_result.png)
