import cv2 as cv                                                              
import matplotlib.pyplot as plt                                              

img1 = cv.imread('mot_color70.jpg')                                           # 1번째,
img2 = cv.imread('mot_color83.jpg')                                           # 2번째 이미지 파일 로드
if img1 is None or img2 is None:                                              # 이미지 로드 실패 시
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")                # 오류

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)                                # 이미지 BGR → RGB 변환
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)                                # 이미지 BGR → RGB 변환

sift = cv.SIFT_create(nfeatures=300)                                          # SIFT 객체 생성 (최대 특징점 300개)

kp1, des1 = sift.detectAndCompute(img1_rgb, mask=None)                        # 첫 번째 이미지 특징점 및 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_rgb, mask=None)                        # 두 번째 이미지 특징점 및 디스크립터 계산
print(f"img1 특징점 수: {len(kp1)},  img2 특징점 수: {len(kp2)}")            # 각 이미지 특징점 개수 출력

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)                                # BFMatcher 생성 (L2 거리, 교차 검사 활성화)
matches = bf.match(des1, des2)                                                # 두 디스크립터 간 1:1 매칭 수행
matches = sorted(matches, key=lambda x: x.distance)                          # 거리 기준 오름차순 정렬 (좋은 매칭 우선)
print(f"매칭 수: {len(matches)}")                                             # 매칭 개수 출력

img_matches = cv.drawMatches(                                                 # 매칭 결과 이미지 생성
    img1_rgb, kp1,                                                            # 첫 번째 이미지와 특징점
    img2_rgb, kp2,                                                            # 두 번째 이미지와 특징점
    matches[:50],                                                             # 상위 50개 매칭만 표시
    outImg=None,                                                              # 출력 이미지 (None이면 새로 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS                          # 매칭되지 않은 특징점은 표시 안 함
)

plt.figure(figsize=(14, 6))                                                   # 그림 크기 설정
plt.imshow(img_matches)                                                       # 매칭 결과 이미지 출력
plt.title(f'BFMatcher Result (Top 50 / {len(matches)})', fontsize=13)        # 제목 설정
plt.axis('off')                                                               # 축 숨기기
plt.tight_layout()                                                            # 여백 자동 조정
plt.savefig('sift_matching_result.png', dpi=150, bbox_inches='tight')         # 결과 이미지 파일로 저장
plt.show()                                                                    # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_matching_result.png")                      # 저장 완료 메시지 출력