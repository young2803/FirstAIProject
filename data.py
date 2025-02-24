import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 데이터 로드 및 전처리

# 아직은 시각화로는 무슨 방법으로 스케일링할 지 판단할 수 없어서 
# 정규화와 표준화 두개 다 만들어서 비교해보기로 함

df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')
df = df.drop('ReadingTime', axis=1)
df = df[(df['SleepTime'] <= 10) & (df['SleepTime'] >= 3)]

feature_columns = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']

# 원본 데이터 먼저 분할
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_columns], df["SleepTime"], test_size=0.2, random_state=42
)

# 정규화 적용
scaler_minmax = MinMaxScaler()
scaler_sleep_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)
y_train_minmax = scaler_sleep_minmax.fit_transform(y_train.values.reshape(-1, 1))
y_test_minmax = scaler_sleep_minmax.transform(y_test.values.reshape(-1, 1))

# 표준화 적용
scaler_standard = StandardScaler()
scaler_sleep_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)
y_train_standard = scaler_sleep_standard.fit_transform(y_train.values.reshape(-1, 1))
y_test_standard = scaler_sleep_standard.transform(y_test.values.reshape(-1, 1))

# 모델 생성 함수
def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train_minmax.shape[1],)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 출력층 (수면시간 예측)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 정규화 모델 학습
model_minmax = build_model()
history_minmax = model_minmax.fit(X_train_minmax, y_train_minmax, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

# 표준화 모델 학습
model_standard = build_model()
history_standard = model_standard.fit(X_train_standard, y_train_standard, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

# 정규화 모델 평가
test_loss_minmax, test_mae_minmax = model_minmax.evaluate(X_test_minmax, y_test_minmax)
print(f"[정규화] 테스트 손실(MSE): {test_loss_minmax:.4f}, 테스트 MAE: {test_mae_minmax:.4f}")

# 표준화 모델 평가
test_loss_standard, test_mae_standard = model_standard.evaluate(X_test_standard, y_test_standard)
print(f"[표준화] 테스트 손실(MSE): {test_loss_standard:.4f}, 테스트 MAE: {test_mae_standard:.4f}")

# 정규화 모델이 더 적합
