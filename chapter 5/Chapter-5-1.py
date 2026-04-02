"""
간단한 이미지 분류기 구현 (MNIST)

요구사항:
1) MNIST 데이터셋 로드
2) 훈련/테스트 세트 분리
3) 간단한 신경망 모델 구성
4) 모델 학습 및 정확도 평가
"""

# TensorFlow/Keras를 사용해 딥러닝 모델을 만들기 위한 모듈들을 가져옵니다.
import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import mnist


# 1. MNIST 데이터셋을 로드합니다.
#    x_train, y_train: 학습용 이미지/정답 라벨
#    x_test, y_test: 테스트용 이미지/정답 라벨
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 이미지 픽셀 값을 정규화합니다.
#    원래 픽셀 범위는 0~255이므로, 255로 나눠 0~1 범위로 스케일링합니다.
#    이렇게 하면 학습이 더 안정적으로 진행됩니다.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 데이터 형태를 출력해 데이터셋이 잘 로드되었는지 확인합니다.
print(f"훈련 이미지 shape: {x_train.shape}, 훈련 라벨 shape: {y_train.shape}")
print(f"테스트 이미지 shape: {x_test.shape}, 테스트 라벨 shape: {y_test.shape}")


# 3. 간단한 신경망(Sequential 모델)을 구성합니다.
#    - Flatten: 28x28 이미지를 784 길이의 1차원 벡터로 펼침
#    - Dense(128, relu): 은닉층
#    - Dense(10, softmax): 0~9 숫자 10개 클래스에 대한 확률 출력층
model = Sequential(
	[
		Flatten(input_shape=(28, 28)),
		Dense(128, activation="relu"),
		Dense(10, activation="softmax"),
	]
)

# 모델 학습 설정(컴파일)
# - optimizer: 가중치 업데이트 방식 (adam)
# - loss: 다중 클래스 분류용 손실 함수
# - metrics: 학습 중 정확도 확인
model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"],
)

# 모델 구조 요약을 출력해 레이어 구성을 확인합니다.
model.summary()


# 4. 모델을 학습시킵니다.
#    epochs=5는 전체 학습 데이터를 5번 반복 학습한다는 의미입니다.
#    validation_split=0.1은 학습 데이터 중 10%를 검증용으로 사용합니다.
history = model.fit(
	x_train,
	y_train,
	epochs=5,
	batch_size=32,
	validation_split=0.1,
)


# 5. 테스트 데이터로 최종 성능(손실, 정확도)을 평가합니다.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# 평가 결과를 출력합니다.
print(f"테스트 손실(loss): {test_loss:.4f}")
print(f"테스트 정확도(accuracy): {test_accuracy:.4f}")


# (선택) 첫 테스트 샘플에 대한 예측 예시
# model.predict는 각 클래스(0~9)에 대한 확률을 반환합니다.
sample_pred = model.predict(x_test[:1], verbose=0)
predicted_label = tf.argmax(sample_pred[0]).numpy()
true_label = y_test[0]
print(f"첫 테스트 이미지 - 실제 라벨: {true_label}, 예측 라벨: {predicted_label}")
