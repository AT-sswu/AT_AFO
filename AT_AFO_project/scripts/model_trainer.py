import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_model(X, y):
    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 훈련 데이터/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 모델
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, validation_split=0.1, verbose=1)

    # 테스트 데이터 -> 모델 평가
    try:
        loss, acc = model.evaluate(X_test, y_test)
        print(f"테스트 정확도: {acc * 100:.2f}%")
    except Exception as e:
        print(f"에러 발생: {e}")

    # 모델 저장
    model.save("models/frequency_classifier.h5")
    return model, scaler
