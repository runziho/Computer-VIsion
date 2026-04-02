import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data() # MNIST 데이터셋을 로드

# 데이터를 훈련 세트와 테스트 세트로 분할
print("훈련 이미지 shape:", x_train.shape)   # (60000, 28, 28) # 훈련 세트
print("테스트 이미지 shape:", x_test.shape)   # (10000, 28, 28) # 데이터 세트

# 데이터 전처리 
x_train = x_train / 255.0 # 픽셀값을 0~255 -> 0~1로 정규화
x_test = x_test / 255.0 

y_train = to_categorical(y_train, 10) # One-Hot 인코딩
y_test = to_categorical(y_test, 10)

#간단한 신경망 모델을 구축
model = Sequential([             # Sequential 모델 활용
    Flatten(input_shape=(28, 28)),   # 28x28 이미지를 1차원으로 펼침
    Dense(128, activation='relu'),   # Dense 레이어를 활용 / 은닉층
    Dense(10, activation='softmax')  # 출력층: 숫자 0~9 총 10개
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)

print("테스트 손실:", test_loss)
print("테스트 정확도:", test_acc)