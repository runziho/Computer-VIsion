# 1. 소벨 에지 검출 및 결과 시각화

- edgeDetectionImage 이미지를 그레이스케일로 변환
- Sobel 필터를 사용하여 x축과 y축의 에지를 검출
- 검출된 에지 강도 이미지를 시각화


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

### 전체 코드
```python
import cv2 as cv             
import numpy as np        
import matplotlib.pyplot as plt

img = cv.imread('dabo.jpg')  # 원본 이미지 불러오기

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일 변환

edges = cv.Canny(gray, 100, 200)  # 캐니 에지 검출(threshold1=100, threshold2=200으로 에지 찾기)

lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)  # 허프 변환으로 직선 검출 (파라미터는 적당히 설정)

line_img = img.copy()  # 원본 복사

# 원본 이미지에 직선 그리기
if lines is not None:  # 검출된 직선이 있을 때만 실행
    for line in lines:  # 각 직선에 대해 반복
        x1, y1, x2, y2 = line[0]  # 직선 좌표 꺼내기
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색(0,0,255), 두께 2로 직선 그리기

plt.figure(figsize=(10,5))  # 전체 화면 크기 설정

plt.subplot(1,2,1)  # 1번째 이미지
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 이미지 출력 (RGB 변환)
plt.title('Original Image')  # 이름
plt.axis('off')  # 축 제거

plt.subplot(1,2,2)  # 2번 이미지
plt.imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB))  # 직선 그려진 이미지 출력
plt.title('Detected Lines')  # 이름
plt.axis('off')  # 축 제거

plt.show()  # 결과 출력
```
## 1) cv.Canny()를 사용하여 에지 맵 생성
```python
edges = cv.Canny(gray, 100, 200)  # 캐니 에지 검출(threshold1=100, threshold2=200으로 에지 찾기)
```
## 2) cv.HoughtLinesP()를 사용하여 직선 검출
```python
lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)  # 허프 변환으로 직선 검출 (파라미터는 적당히 설정)
```
## 3) cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림
```python
# 원본 이미지에 직선 그리기
if lines is not None:  # 검출된 직선이 있을 때만 실행
    for line in lines:  # 각 직선에 대해 반복
        x1, y1, x2, y2 = line[0]  # 직선 좌표 꺼내기
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색(0,0,255), 두께 2로 직선 그리기
```
## 4) Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
```python
plt.subplot(1,2,1)  # 1번째 이미지
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 이미지 출력 (RGB 변환)
plt.title('Original Image')  # 이름
plt.axis('off')  # 축 제거

plt.subplot(1,2,2)  # 2번 이미지
plt.imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB))  # 직선 그려진 이미지 출력
plt.title('Detected Lines')  # 이름
plt.axis('off')  # 축 제거

plt.show()  # 결과 출력
```

## 실행 결과
<img width="1499" height="833" alt="image" src="https://github.com/user-attachments/assets/80cedac9-c235-4e9f-a36a-fe161a4f510d" />

# 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출
- coffee cup 이미지로 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체 추출
- 객체 추출 결과를 마스크 형태로 시각화
- 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력

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
## 1) cv.grabCut을 이용하여 대화식 분할을 수행
```python
mask = np.zeros(img.shape[:2], np.uint8)  # 이미지 크기와 동일한 마스크 생성 (초기값 0)
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델 초기화
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델 초기화

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)  # 사용자가 지정한 영역(rect)을 기준으로 GrabCut 수행 (배경/객체 분리)
```

## 2) 초기 사각형 영역은 (x,y,width,height) 형식으로 설정
```python
rect = cv.selectROI("Select Object", img, False)  # (x, y, width, height) 형태로 자동 생성
```

## 3) 마스크를 사용하여 원본 이미지에서 배경을 제거
```python
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')  # 배경(0), 객체(1)로 마스크 재구성

result = img * mask2[:, :, np.newaxis]  # 마스크를 이용해 객체 부분만 남기고 배경 제거
```

## 4) matplotlib를 사용하여 원본, 마스크, 배경 제거 이미지 세 개를 나란히 시각화
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

## 출력 결과
- 실행시 화면
  <img width="1916" height="1492" alt="image" src="https://github.com/user-attachments/assets/ab49ecf7-a2c5-428b-80e4-25df2bd8bdb5" />


` ROI 지정 
<img width="1912" height="1473" alt="image" src="https://github.com/user-attachments/assets/df9104e2-df38-48e7-ab60-ed92c9f33534" />

- 최종 출력
- <img width="2245" height="852" alt="image" src="https://github.com/user-attachments/assets/e9a97581-14ba-452b-a0e8-f734b4ef4480" />

