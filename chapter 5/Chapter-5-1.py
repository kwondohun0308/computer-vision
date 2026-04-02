"""
2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

요구사항:
1) CIFAR-10 데이터셋 로드
2) 데이터 전처리(정규화)
3) CNN 모델 설계 및 훈련
4) 모델 성능 평가
5) 테스트 이미지(dog.jpg) 예측
"""

# 파일 경로 확인을 위해 os 모듈을 사용합니다.
import os

# TensorFlow/Keras 관련 모듈을 가져옵니다.
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.utils import load_img, img_to_array
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# CIFAR-10 클래스 이름(인덱스 0~9)을 사람이 읽을 수 있는 문자열로 매핑합니다.
class_names = [
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
]


# 1) CIFAR-10 데이터셋 로드
#    x_train, y_train: 학습 데이터
#    x_test, y_test: 테스트 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 라벨은 (N, 1) 형태이므로 (N,)으로 변환해 손실 함수에서 다루기 쉽게 만듭니다.
y_train = y_train.squeeze()
y_test = y_test.squeeze()

print(f"훈련 이미지 shape: {x_train.shape}, 훈련 라벨 shape: {y_train.shape}")
print(f"테스트 이미지 shape: {x_test.shape}, 테스트 라벨 shape: {y_test.shape}")


# 2) 데이터 전처리 (정규화)
#    픽셀 값 범위(0~255)를 0~1로 변환해 학습 수렴을 빠르게 합니다.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 3) CNN 모델 설계
#    Conv2D + MaxPooling2D 블록을 쌓아 특징을 추출하고,
#    Flatten + Dense로 최종 분류를 수행합니다.
#    데이터 증강(RandomFlip/Rotation/Zoom), BatchNormalization, Dropout을 추가해
#    일반화 성능과 테스트 정확도를 높입니다.
data_augmentation = Sequential(
    [
        # 첫 번째 레이어에 input_shape 추가!
        RandomFlip("horizontal", input_shape=(32, 32, 3)), 
        RandomRotation(0.1),
        RandomZoom(0.1),
    ],
    name="data_augmentation",
)

model = Sequential(
	[
		data_augmentation,
		Conv2D(32, (3, 3), padding="same", activation="relu"),
		BatchNormalization(),
		Conv2D(32, (3, 3), padding="same", activation="relu"),
		BatchNormalization(),
		MaxPooling2D((2, 2)),
		Dropout(0.25),
		Conv2D(64, (3, 3), padding="same", activation="relu"),
		BatchNormalization(),
		Conv2D(64, (3, 3), padding="same", activation="relu"),
		BatchNormalization(),
		MaxPooling2D((2, 2)),
		Dropout(0.3),
		Conv2D(128, (3, 3), padding="same", activation="relu"),
		BatchNormalization(),
		Conv2D(128, (3, 3), padding="same", activation="relu"),
		BatchNormalization(),
		MaxPooling2D((2, 2)),
		Dropout(0.4),
		Flatten(),
		Dense(256, activation="relu"),
		BatchNormalization(),
		Dropout(0.5),
		Dense(10, activation="softmax"),
	]
)

# 모델 학습 설정
# - optimizer: Adam
# - loss: sparse_categorical_crossentropy (정수 라벨에 적합)
# - metrics: 정확도
model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"],
)

# 모델 구조를 출력해 레이어 구성이 올바른지 확인합니다.
model.summary()


# 4) 모델 훈련
#    validation_split=0.1: 학습 데이터의 10%를 검증용으로 사용
history = model.fit(
	x_train,
	y_train,
	epochs=30,
	batch_size=64,
	validation_split=0.1,
	callbacks=[
		EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
		ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
	],
)

# 테스트 데이터로 최종 성능 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"테스트 손실(loss): {test_loss:.4f}")
print(f"테스트 정확도(accuracy): {test_accuracy:.4f}")


# 5) dog.jpg 예측 (중앙 크롭 버전)
image_path = "dog.jpg"

if os.path.exists(image_path):
    # 1. 원본 비율 그대로 일단 불러옵니다.
    img = load_img(image_path)
    img_array = img_to_array(img)

    # 2. 비율이 찌그러지지 않게, 가운데 '정사각형' 부분만 예쁘게 잘라냅니다. (Center Crop)
    h, w = img_array.shape[0], img_array.shape[1]
    min_dim = min(h, w)
    
    # 가로/세로 중 작은 치수를 기준으로 중앙을 잡습니다.
    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2
    cropped_array = img_array[start_y : start_y + min_dim, start_x : start_x + min_dim]

    # 3. 잘라낸 정사각형을 32x32 해상도로 축소하고 정규화합니다.
    resized_array = tf.image.resize(cropped_array, [32, 32]).numpy()
    final_array = resized_array.astype("float32") / 255.0

    # 테스트 타임 증강(TTA) 추론
    batch_original = tf.expand_dims(final_array, axis=0)
    batch_flip = tf.expand_dims(tf.image.flip_left_right(final_array), axis=0)
    
    probs_original = model.predict(batch_original, verbose=0)[0]
    probs_flip = model.predict(batch_flip, verbose=0)[0]
    pred_probs = (probs_original + probs_flip) / 2.0

    pred_index = int(tf.argmax(pred_probs).numpy())
    pred_label = class_names[pred_index]
    confidence = float(pred_probs[pred_index])

    top3_indices = tf.argsort(pred_probs, direction="DESCENDING")[:3].numpy()

    print("\n🐾 dog.jpg 예측 결과")
    print(f"예측 클래스: {pred_label}")
    print(f"신뢰도: {confidence:.4f}")
    print("상위 3개 클래스 예측:")
    for idx in top3_indices:
        print(f"- {class_names[int(idx)]}: {float(pred_probs[int(idx)]):.4f}")
else:
    print("dog.jpg 파일을 찾을 수 없습니다.")
