import cv2 as cv                                                        
import matplotlib.pyplot as plt                                         

img = cv.imread('mot_color70.jpg')                                        # 이미지 파일 읽기
if img is None:                                                           # 이미지 로드 실패 시
    raise FileNotFoundError("이미지를 찾을 수 없습니다: mot_color70.jpg") # 오류 

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)                              # BGR → RGB 변환 (matplotlib용)

sift = cv.SIFT_create(                                                    # SIFT 객체 생성
    nfeatures=300,                                                        # 검출할 최대 특징점 수
    nOctaveLayers=3,                                                      # 옥타브당 레이어 수 (기본값 3)
    contrastThreshold=0.04,                                               # 낮을수록 특징점 많아짐 (기본값 0.04)
    edgeThreshold=10,                                                     # 높을수록 엣지 부근 특징점 많아짐 (기본값 10)
    sigma=1.6                                                             # 가우시안 블러 시그마 (기본값 1.6)
)

keypoints, descriptors = sift.detectAndCompute(img_rgb, mask=None)        # 특징점 및 128차원 디스크립터 계산
print(f"검출된 특징점 수: {len(keypoints)}")                              # 검출된 특징점 개수 출력

img_keypoints = cv.drawKeypoints(                                         # 특징점을 이미지에 시각화
    img_rgb,                                                              # 원본 이미지
    keypoints,                                                            # 검출된 특징점 리스트
    outImage=None,                                                        # 출력 이미지 (None이면 새로 생성)
    color=(0, 255, 0),                                                    # 특징점 색상 (초록색)
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS                       # 특징점 위치 + 크기 + 방향 모두 표시
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))                           # 1행 2열 서브플롯 생성
fig.suptitle('SIFT Feature Detection', fontsize=16, fontweight='bold')    # 전체 제목 설정

axes[0].imshow(img_rgb)                                                   # 1번 칸에 원본 이미지 출력
axes[0].set_title('Original Image', fontsize=13)                          # 1번 칸 칸 제목 설정
axes[0].axis('off')                                                       # 1번 칸 축 숨기기

axes[1].imshow(img_keypoints)                                             # 2번 칸에 특징점 이미지 출력
axes[1].set_title(f'SIFT Keypoints  (n = {len(keypoints)})', fontsize=13) # 2번 칸 제목 설정
axes[1].axis('off')                                                       # 2번 칸 축 숨기기

plt.tight_layout()                                                        # 서브플롯 간격 자동 조정
plt.savefig('sift_result.png', dpi=150, bbox_inches='tight')              # 결과 이미지 파일로 저장
plt.show()                                                                # 화면에 이미지 출력
print("결과 이미지 저장 완료: sift_result.png")                           # 저장 완료 메시지 출력