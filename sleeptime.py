import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import json

# 테스트 횟수
# ex) start = 1, end = 8, 모델 훈련 8번 반복, 결과 1 ~ 8번으로 저장
start = 1
end = 8

# 데이터셋 로드
df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')

# 모델 훈련에 사용할 데이터 분류
# features로 지정한 데이터들이 label 데이터와 어떤 상관관계를 이루는지 분석
# ********************************************************************
# [Features]
# WortoutTime: 운동 시간
# PhoneTime: 업무 외 전자기기 사용 시간
# WorkHours: 근무 or 공부 시간
# CaffeineIntake: 일일 카페인 섭취량, 단위: mg(밀리그램)
# RelaxationTime: 휴식 시간(전자기기 사용 X)
# ********************************************************************
# [Label]
# SleepTime: 수면 시간
features = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
label = ['SleepTime']

# 데이터 스케일링, Min-Max 정규화 활용
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# 훈련 데이터와 테스트 데이터로 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.2, random_state=42)

# 하이퍼파라미터 설정
batch_sizes = [16, 32, 64, 128]
learning_rates = [0.001, 0.01, 0.1]
l2_alpha = [0.0001, 0.001, 0.01]

best_mse = float('inf')
best_params = {}

# JSON 파일 저장 경로
json_file_path = 'recode_smallest_mse_and_params.json'

# 결과를 저장할 리스트
results = []

# 위에서(start, end) 지정한 횟수대로 반복 실행
for i in range(start, end+1):
    best_mse = float('inf')
    best_params = {}

    def model_test():
        global best_mse, best_params  # 함수 내에서 수정 가능하도록 global 선언

        # 하이퍼파라미터를 다양하게 조합하며 모델 훈련
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for al in l2_alpha:
                    # 현재 모델 훈련에 적용된 하이퍼파라미터 조합 출력, 실행 속도를 높이기 위해 실행 시 주석 처리 권장
                    # print(f'[{i}] Training with batch_size={batch_size}, lr={lr}, al={al}')
                    
                    # 이전 훈련 모델 초기화
                    tf.keras.backend.clear_session()

                    # 훈련에 사용할 모델 정의
                    # Dropout은 chatGPT 권고에 따라 0.2~0.5 사이의 값 중 임의로 지정
                    # 활성화 함수로 relu 사용, l2 규제 적용
                    # chatGPT 권고에 따라 128 -> 64 -> 32 -> 1로 Dense가 점점 줄어드는 형태로 구성
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],), kernel_regularizer=l2(al)),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(al)),
                        tf.keras.layers.Dropout(0.4),
                        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(al)),
                        tf.keras.layers.Dropout(0.25),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])

                    # 모델 컴파일, Optimizer로는 Adam 고정, 손실 함수로 mse 사용
                    optimizer = Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                    # 불필요한 학습 시간을 줄이기 위해 EarlyStopping 적용
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    # 모델 훈련
                    history = model.fit(x_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[early_stopping])

                    # 테스트 데이터를 기반으로 예측
                    # 예측한 mse를 변수에 저장
                    y_pred = model.predict(x_test).flatten()
                    mse = mean_squared_error(y_test, y_pred)

                    # 저장된 mse가 best_mse 보다 작다면 best_mse, best_params(현 하이퍼파라미터) 갱신
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {'batch_size': batch_size, 'learning_rate': lr, 'l2_alpha': al}

        # 일련의 과정이 종료된 후 얻은 가장 최고의 mse, params를 반복 횟수와 함께 results에 저장
        result = {
            'iteration': i,
            'best_mse': best_mse,
            'best_params': best_params
        }
        results.append(result)

    # 실행
    model_test()

    # JSON 파일에 한 줄씩 저장
    with open(json_file_path, 'a') as f:
        f.write(json.dumps(results[-1]) + '\n')

print(f'모든 결과가 {json_file_path} 파일에 저장되었습니다.')