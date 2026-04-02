import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
# 입력 이미지의 픽셀값을 0~1 범위로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# CIFAR-10 클래스 이름 정의
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
# 다중 클래스 분류 문제이므로 sparse_categorical_crossentropy 사용
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# 테스트 데이터셋을 이용한 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("테스트 손실:", test_loss)
print("테스트 정확도:", test_acc)

# 외부 이미지(dog.jpg) 예측
img = load_img("dog.jpg", target_size=(32, 32))
img_array = img_to_array(img)

# 학습 데이터와 동일한 방식으로 정규화 수행
img_array = img_array / 255.0

# 모델 입력 형태에 맞게 배치 차원 추가
img_array = np.expand_dims(img_array, axis=0)

# 예측 수행
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

print("예측 결과:", class_names[predicted_class])
print("각 클래스별 확률:", prediction)