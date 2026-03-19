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