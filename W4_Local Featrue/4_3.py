import cv2 as cv                                                                      
import matplotlib.pyplot as plt                                                    
import numpy as np                                                                   

img1 = cv.imread('img1.jpg')   # img 1 로드
img2 = cv.imread('img2.jpg')   # img2 로드
if img1 is None or img2 is None:                                                       # 이미지 로드 실패 시
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")                         # 오류
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)                                         # img 1 BGR → RGB 변환
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)                                         # img 2 BGR → RGB 변환

sift = cv.SIFT_create()                                                                # SIFT 객체 생성

kp1, des1 = sift.detectAndCompute(img1_rgb, mask=None)                                 # img1의 특징점 및 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_rgb, mask=None)                                 # img2의 특징점 및 디스크립터 계산
print(f"img1 특징점 수: {len(kp1)},  img2 특징점 수: {len(kp2)}")                     # 각 이미지 특징점 개수 출력

bf = cv.BFMatcher()                                                                    # BFMatcher 객체 생성 (기본: L2 거리)
knn_matches = bf.knnMatch(des1, des2, k=2)                                             # 각 특징점에 대해 최근접 이웃 2개 매칭

good_matches = []                                                                      # 좋은 매칭 결과를 저장할 리스트
for m, n in knn_matches:                                                               # 매칭 쌍 순회
    if m.distance < 0.7 * n.distance:                                                  # Lowe's ratio test: 거리 비율이 0.7 미만이면
        good_matches.append(m)                                                         # 좋은 매칭으로 채택
print(f"좋은 매칭 수: {len(good_matches)}")                                            # 좋은 매칭 개수 출력

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])                         # img1의 매칭 특징점 좌표 추출
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])                         # img2의 매칭 특징점 좌표 추출

H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)                               # RANSAC으로 호모그래피 행렬 계산 (이상점 제거)
print(f"호모그래피 행렬:\n{H}")                                                        # 계산된 호모그래피 행렬 출력

h1, w1 = img1_rgb.shape[:2]                                                            # img1 높이·너비 추출
h2, w2 = img2_rgb.shape[:2]                                                            # img2 높이·너비 추출
panorama_w = w1 + w2                                                                   # 파노라마 너비: 두 이미지 너비 합산
panorama_h = max(h1, h2)                                                               # 파노라마 높이: 두 이미지 중 큰 높이 사용

warped = cv.warpPerspective(img2_rgb, H, (panorama_w, panorama_h))                    # img2를 호모그래피로 변환하여 img1 시점에 정렬
warped[0:h1, 0:w1] = img1_rgb                                                          # 변환된 이미지 위에 img1을 덮어씌워 합성

match_draw = cv.drawMatches(                                                           # 매칭 결과 이미지 생성
    img1_rgb, kp1,                                                                     # img2와 특징점
    img2_rgb, kp2,                                                                     # img2와 특징점
    good_matches[:50],                                                                 # 상위 50개 좋은 매칭만 표시
    outImg=None,                                                                       # 출력 이미지 (None이면 새로 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS                                   # 매칭되지 않은 특징점은 표시 안 함
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))                                        # 1행 2열 서브플롯 생성
fig.suptitle('SIFT Homography', fontsize=16, fontweight='bold')                        # 전체 제목 설정

axes[0].imshow(match_draw)                                                             # 1번 칸에 매칭 결과 출력
axes[0].set_title(f'Matching Result ({len(good_matches)} matches)', fontsize=13)       # 1번째 칸 제목 설정
axes[0].axis('off')                                                                    # 1번째 칸 축 숨기기

axes[1].imshow(warped)                                                                 # 2번 칸에 변환·합성된 이미지 출력
axes[1].set_title('Warped Image (Panorama)', fontsize=13)                              # 2번째 칸 제목 설정
axes[1].axis('off')                                                                    # 2번째 칸 축 숨기기

plt.tight_layout()                                                                     # 서브플롯 간격 자동 조정
plt.savefig('sift_homography_result.png', dpi=150, bbox_inches='tight')                # 결과 이미지 파일로 저장
plt.show()                                                                             # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_homography_result.png")                             # 저장 완료 메시지 출력