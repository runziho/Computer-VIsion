import cv2 as cv
import sys #cv와 sys import
import numpy as np   # 요구사항인 np.hstack 사용하려면 필요

img = cv.imread('soccer.jpg') #cv.imread()를 사용하여 이미지 로드

if img is None : 
    sys.exit('파일이 존재하지 않습니다.') #img파일이 없다면 종료

#그레이스케일 변환
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
cv.imwrite('soccer_gray.jpg',gray) #변환 영상을 파일에 저장

gray_soccer = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) #흑백픽셀을 컬러 형식으로 변환

hstack = np.hstack((img, gray_soccer)) #두 사진 가로로 연결

cv.imshow('Result', hstack) #연결된 사진 출력
cv.waitKey(0) #입력이 들어올 때까지 기다림
cv.destroyAllWindows() #Open cv가 만든 창을 완전히 닫음
