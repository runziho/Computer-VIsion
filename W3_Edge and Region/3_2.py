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