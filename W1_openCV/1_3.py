import cv2 as cv
import sys

img = cv.imread('girl_laughing.jpg') #img 로드

if img is None : 
    sys.exit('파일이 존재하지 않습니다.') #img파일이 없다면 종료

imgs = img.copy() #원본용 이미지 저장해두기

ix, iy = None, None #시작 좌표 
drawing = False #드래그 중 여부
roi = None #선택된 영역

def draw(event,x,y,flags,param):#마우스 이벤트 함수
    global ix,iy,drawing,img,roi #전역변수

    if event == cv.EVENT_LBUTTONDOWN: #마우스 왼쪽 버튼 클릭했을 떄 초기 위치 저장
        drawing = True #드래그 상태 
        ix,iy = x,y #현재 위치를 시작 좌표로 저장
    
    elif event == cv.EVENT_MOUSEMOVE: #드래그 중이면 사각형 표시
        if drawing:
            img = imgs.copy() #계속 새로 그림
            cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2) #사각형 그림
    
    elif event == cv.EVENT_LBUTTONUP: #마우스 떼면 ROI 선택된것 별도의 창에 출력
        drawing = False
        img = imgs.copy() #계속 새로 그림
        cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2) #사각형 그리기

        #numpy 슬라이싱으로 roi 추출
        x1,x2 = min(ix,x), max(ix,x)
        y1,y2 = min(iy,y), max(iy,y)

        roi = imgs[y1:y2, x1:x2] #numpy 슬라이싱을 사용하여 roi 저장

        if roi.size !=0:
            cv.imshow('ROI',roi) #roi가 존재하면 (0이아니면) ROI창 띄움

    cv.imshow('Drawing',img)

cv.namedWindow('Drawing') #윈도우 이름 Drawing로 선언  
cv.setMouseCallback('Drawing', draw) #마우스 이벤트 처리

while True: #계속 반복
    cv.imshow('Drawing', img) #Drawing 창 만들기

    key = cv.waitKey(1) & 0xFF   #키 입력 받기

    if key == ord('r'): # r 입력받을 시 선택 리셋
        img = imgs.copy()
        cv.destroyWindow('ROI')  # ROI 창 닫기

    elif key == ord('s'): #s입력받으면 ROI img 저장
        if roi is not None:
            cv.imwrite('roi.jpg', roi) #ROI 저장
            print('ROI saved') #ROI 저장 완료 출력

    elif key == ord('q'): # q 입력시 종료
        break

cv.destroyAllWindows() #Open cv가 만든 창을 완전히 닫음
