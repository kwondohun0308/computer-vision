# chapter 5
## 과제1 설명 및 요구사항 (간단한 이미지 분류기 구현)
 - 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현
 - MNIST 데이터셋을 로드
 - 데이터를 훈련 세트와 테스트 세트로 분할
 - 간단한 신경망 모델을 구축
 - 모델을 훈련시키고 정확도를 평가

과제 한줄 요약 - TensorFlow와 Keras를 활용해, 0부터 9까지의 손글씨 이미지를 보고 컴퓨터가 스스로 패턴을 학습하여 어떤 숫자인지 확률적으로 맞춰내는 '인공신경망'을 구축하는 과정

<details>
	<summary>과제 1 전체 코드</summary>
		
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



</details>

![1번 결과](https://github.com/user-attachments/assets/5f1be9b1-881c-4d78-8343-ea9dc221b4c9)


## 과제 1 주요 코드 설명
1. 데이터 전처리

	```python
		x_train = x_train.astype("float32") / 255.0
		x_test = x_test.astype("float32") / 255.0
	```
	가장 먼저 정규화를 시작합니다. 0부터 255까지의 픽셀값들을 255로 나누어 0과 1사이의 작은 소수점 숫자로 압축합니다. 이렇게 데이터를 줄여주면 인공지능이 훨씬 바르고 안정적으로 학습할 수 있습니다.


2. 신경망 모델 구조 설계
   ```python
		model = Sequential(
			[
				Flatten(input_shape=(28, 28)),
				Dense(128, activation="relu"),
				Dense(10, activation="softmax"),
			]
		)
   ```
   Sequential: "지금부터 신경망을 순차적으로 쌓아 올리겠습니다"라는 선언

   Flatten (입력층): 28x28 형태의 정사각형 이미지(2D)를 모델이 읽기 편하도록 784칸짜리 긴 1차원 줄(1D)로 변환해줍니다.

   Dense(128, relu) (은닉층): 128개의 뉴런이 이미지의 핵심 특징을 추출하고, relu 필터로 불필요한 마이너스 신호를 걸러내 확실한 정보만 다음으로 전달
   (128개는 이미지의 복잡도를 고려하여 실험적으로 설정했으며, 뉴런수가 1024처럼 너무 크게 늘리면 불필요한 노이즈까지 모두 학습해서 오히려 정확도가 떨어지는 과적합 현상이 일어납니다.)

   Dense(10, softmax) (출력층): 맞춰야 할 숫자가 0부터 9까지 10개이므로, 최종 값도 10개로 만듭니다. softmax는 '이 숫자가 7일 확률 90%, 3일 확률 5%, 1일 확률 5%...'처럼 총합이 100%가 되는 확률 값으로 변환해 줍니다.

3. 컴파일 (학습 규칙 설정)
   ```python
		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"],
		)	
   ```
   optimizer="adam" - 정답을 향해 찾아가는 길 찾기 알고리즘입니다. 파라미터별로 학습 보폭을 스스로 조절하며, 별도의 복잡한 튜닝 없이도 빠르고 안정적인 수렴이 보장

  loss="sparse_categorical_crossentropy" -  모델이 예측한 값과 실제 정답이 얼마나 차이 나는지 채점하는 기준입니다. 이 오차가 0에 가까워지는 방향으로 모델은 스스로를 수정(가중치 업데이트)해 나갑니다.

  일반적인 categorical_crossentropy를 쓰려면 정답 데이터를 일일이 [0,0,1,0...] 형태의 one-hot 인코딩으로 변환해 주어야 하지만, 저희가 로드한 MNIST 정답 데이터는 0부터 9까지의 정수형으로 되어 있기 때문에, 데이터 변환 과정을 생략하고 메모리를 효율적으로 아끼기 위해 정수형 정답지를 바로 처리할 수 있는 Sparse Categorical Crossentropy를 사용했습니다.

4. 모델 학습(Training)
   ```python
		history = model.fit(
			x_train,
			y_train,
			epochs=5,
			batch_size=32,
			validation_split=0.1,
		)
   ```
   epochs=5 - 반복학습으로 준비된 training 데이터를 처음부터 끝까지 5번 반복해서 수행

   batch_size=32: 학습할때 한꺼번에 보지 않고, 한 번에 32개씩 쪼개서 학습하겠다는 뜻

   validation_split=0.1 - 준비된 데이터를 10%정도 남겨두고 모델이 답만 단순히 외워버린건 아닌지(과적합 방지), 진짜로 학습하고 있는건지 실시간으로 검증합니다.

## 과제2 설명 및 요구사항 (CIFAR-10 데이터셋을 활용한 CNN 모델 구축)
 - CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행
 - CIFAR-10 데이터셋을 로드
 - 데이터 전처리(정규화 등)를 수행
 - CNN 모델을 설계하고 훈련
 - 모델의 성능을 평가하고, 테스트 이미지(dog.jpg)에 대한 예측을 수행

과제 한줄 요약 - CIFAR-10 데이터셋을 기반으로 CNN(합성곱 신경망)을 학습시켜 이미지 분류 모델을 구축하고, 실제 외부 이미지(dog.jpg)로 실전 예측까지 수행하는 전체 딥러닝 파이프라인 구현 과제

<details>
	<summary>과제 2 전체 코드</summary>
	
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



</details>

![2번 결과](https://github.com/user-attachments/assets/73324de4-18cc-4a68-a083-4757b3a5b05d)


## 과제 2 주요 코드 설명

1. 데이터 증강 (Data Augmentation)
	```python
   		data_augmentation = Sequential(
		    [
		        # 첫 번째 레이어에 input_shape 추가!
		        RandomFlip("horizontal", input_shape=(32, 32, 3)), 
		        RandomRotation(0.1),
		        RandomZoom(0.1),
		    ],
		    name="data_augmentation",
		)
 	```
	가장먼저 데이터 증강층을 통과시킵니다. 사진을 원본 그대로만 보여주지 않고, 좌우로 뒤집고 살짝 회전시키고 확대해서 모델에게 보여줍니다. 이를 통해 모델이 특정 픽셀 위치만 외우는 과적합을 방지하고 일반화 성능을 크게 높였습니다.

2. CNN 모델 설계 (Conv2D & MaxPooling)
	```python
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
 	```
	합성곱 신경망입니다. Conv2D는 마치 돋보기처럼 이미지를 훑으며 선, 색상, 질감 같은 시각적 특징을 찾아냅니다.

	그리고 MaxPooling을 통해 가장 강렬한 특징만 남기고 데이터의 크기를 반으로 줄여버립니다.

	여기에 Dropout을 적용해 학습 시 인공 뉴런의 일부를 무작위로 꺼버림으로써, 모델이 소수의 뉴런에만 의존하지 않고 전체적으로 튼튼하게 학습되도록 설계했습니다."

3. 스마트 학습 매니저 (Callbacks)
	```python
			callbacks=[
				EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
				ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
			],
 	```
	에폭(Epoch)을 30회로 넉넉하게 주었지만, 무식하게 끝까지 돌지 않습니다. EarlyStopping을 통해 성적이 7번 연속 오르지 않으면 가장 똑똑했던 시점으로 가중치를 복구하고 학습을 조기 종료합니다.

	또한 ReduceLROnPlateau를 사용해 정답 근처에서 더 이상 오차가 줄지 않으면 보폭(학습률)을 절반으로 줄여 아주 세밀하게 최적점을 탐색하도록 도입했습니다.

4. 실전 전처리 (Center Crop & TTA)
	```python
 			# 가로/세로 중 작은 치수를 기준으로 중앙을 잡습니다.
		    start_y = (h - min_dim) // 2
		    start_x = (w - min_dim) // 2
		    cropped_array = img_array[start_y : start_y + min_dim, start_x : start_x + min_dim]

			probs_original = model.predict(batch_original, verbose=0)[0]
		    probs_flip = model.predict(batch_flip, verbose=0)[0]
		    pred_probs = (probs_original + probs_flip) / 2.0
 			
 	```
	dog.png 이미지를 강제로 32x32로 압축하면, 강아지가 찌그러지며 마치 날개를 편 새(Bird) 혹은 고양이처럼 인식되는 오류가 있었습니다.

	이를 해결하기 위해 원본의 비율을 유지하며 중앙만 잘라내는 Center Crop 기법을 적용했습니다.

	또한, 예측 단계에서도 원본과 좌우 반전 이미지를 동시에 예측하고 평균을 내는 TTA(Test Time Augmentation) 기법을 적용해, AI가 최종 결정을 내릴 때 한 번 더 신중하게 판단하도록 정확도를 끌어올렸습니다.

