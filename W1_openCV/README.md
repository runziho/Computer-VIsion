# 1. 이미지 불러오기 및 그레이스케일 변환

OpenCV를 사용하여 이미지를 불러오고 화면에 출력
원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시

## 1) cv.imread()를 사용하여 이미지 로드

```python
img = cv.imread('soccer.jpg')
```

## 2) cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환

``` python
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
```
원본 이미지

![soccer](soccer.jpg)
변환 결과 

![soccer_gray](soccer_gray.jpg)

## 3) np.hstack 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력

```python
hstack = np.hstack((img, gray_soccer)) #두 사진 가로로 연결
```

## 4) cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무 키나 누르면 창이 닫히도록 할 것

```python
cv.imshow('Result', hstack) #연결된 사진 출력
cv.waitKey(0) #입력이 들어올 때까지 기다림
cv.destroyAllWindows() #Open cv가 만든 창을 완전히 닫음
```
실행 결과 
<img width="1819" height="1081" alt="image" src="https://github.com/user-attachments/assets/6eaba4ec-a031-4284-b2dd-d9464a989d68" />


