# 1. 소벨 에지 검출 및 결과 시각화

- edgeDetectionImage 이미지를 그레이스케일로 변환
- Sobel 필터를 사용하여 x축과 y축의 에지를 검출
- 검출된 에지 강도 이미지를 시각화

### 그레이스케일 개념


### 전체 코드
```python
import cv2 as cv           
import numpy as np          
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')  # 이미지 파일 불러오기 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # Sobel 필터를 사용하여 x축 에지 검출
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # Sobel 필터를 사용하여 y축 에지 검출

magnitude = cv.magnitude(sobelx, sobely)  # x,y 에지 합쳐서 전체 에지 강도 계산

magnitude = cv.convertScaleAbs(magnitude)  # 보기 좋게 0~255로 변환

plt.figure(figsize=(10,5))  # 전체 출력 화면 크기 설정

plt.subplot(1,2,1)  # 1번 이미지
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 출력 (그레이스케일 변환한거 컬러로 되돌림)
plt.title('Original Image')  # 이름
plt.axis('off')  # 축 제거

plt.subplot(1,2,2)  # 2번 이미지
plt.imshow(magnitude, cmap='gray')  # 에지 이미지를 흑백으로 출력
plt.title('Edge Magnitude')  # 이름
plt.axis('off')  # 축 제거

plt.show()  # 결과 화면에 출력
```

## 1) cv.imread()를 사용하여 이미지 로드

```python
img = cv.imread('edgeDetectionImage.jpg')  # 이미지 파일 불러오기 
```

## 2) cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환

``` python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
```

## 3) svSobel()을 사용하여 x축(cv.CV_64F,1,0)과 y축(cv.CV_64F,0,1) 방향의 에지를 검출

```python
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # Sobel 필터를 사용하여 x축 에지 검출
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # Sobel 필터를 사용하여 y축 에지 검출

```

## 4) cv.magnitude()를 사용하여 에지 강도 계산 

```python
magnitude = cv.magnitude(sobelx, sobely)  # x,y 에지 합쳐서 전체 에지 강도 계산
```
## 5) Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화

```python
plt.subplot(1,2,1)  # 1번 이미지
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 출력 (그레이스케일 변환한거 컬러로 되돌림)
plt.title('Original Image')  # 이름
plt.axis('off')  # 축 제거

plt.subplot(1,2,2)  # 2번 이미지
plt.imshow(magnitude, cmap='gray')  # 에지 이미지를 흑백으로 출력
plt.title('Edge Magnitude')  # 이름
plt.axis('off')  # 축 제거

plt.show()  # 결과 화면에 출력
```
### 실행 결과 
<img width="1493" height="834" alt="image" src="https://github.com/user-attachments/assets/1b197576-4810-4af1-8e6d-8e2451939538" />

# 2. 캐니 에지 및 허프 변환을 이용한 직선 검출
- dabo 이미지에 캐니 에지 검출을 사용하여 에지 맵 생성
- 허프 변환을 사용하여 이미지에서 직선 검출
- 검출된 직선을 원본 이미지에서 빨간색으로 표시

## 1) cv.Canny()를 사용하여 에지 맵 생성

## 2) cv.HoughtLinesP()를 ㅎ
ㅅ
