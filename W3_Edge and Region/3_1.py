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