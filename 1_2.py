import cv2 as cv
import sys

img = cv.imread('soccer.jpg')  # img 불러오기

if img is None:
    sys.exit('파일이 존재하지 않습니다.') #img 없을 시 종료

brush_size = 5  #초기 붓 크기 = 5로 설정

def draw(event, x, y, flags, param): #마우스 이벤트 함수 정의
    global img, brush_size #전역 변수 사용
 
    if event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON): #마우스 이동중이고 좌측버튼 누른 상태일 때
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1) #파란색 원 그림

    elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_RBUTTON): #마우스 이동중이고 우측버튼 누른 상태일 때
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1) #빨간색 원 그림


cv.namedWindow('Painting') #윈도우 이름 
### imshow()가 이미 창을 만드는데 왜 nameWimdow를 쓰는지.... -> setMouseCallback은 “이미 존재하는 창”에 연결 , 따라서 아직 창이 존재하지 않으면 연결 실패 가능성이 있음
cv.setMouseCallback('Painting', draw) #마우스 이벤트 처리

while True: #계속 반복
    cv.imshow('Painting', img) #painting 창 만들기

    key = cv.waitKey(1) & 0xFF   #키 입력 받기
    if key == ord('+'): # + 입력받을 시
        brush_size += 1 #원 크기 +1
        if brush_size > 15: #크기가  15 초과라면
            brush_size = 15 #크기 15로 저장

    elif key == ord('-'): #-입력받으면
        brush_size -= 1 #원 크기 -1
        if brush_size < 1: #크기 1보다 작아지면
            brush_size = 1 #크기 1로 저장

    elif key == ord('q'): # q 입력시 종료
        break

cv.destroyAllWindows() #Open cv가 만든 창을 완전히 닫음