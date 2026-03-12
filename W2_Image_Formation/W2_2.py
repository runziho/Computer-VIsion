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