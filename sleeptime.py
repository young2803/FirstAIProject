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
count = 1

# Load dataset
df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')
df = df.drop('ReadingTime', axis=1)
df = df[(df['SleepTime'] <= 10) & (df['SleepTime'] >= 3)]

# Define features and label
features = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
label = ['SleepTime']

# Normalize feature data
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Split data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.2, random_state=42)

# Hyperparameters
batch_sizes = [16, 32, 64, 128]
learning_rates = [0.001, 0.01, 0.1]
l2_alpha = [0.0001, 0.001, 0.01]

best_mse = float('inf')
best_params = {}

# JSON 파일 저장 경로
json_file_path = 'model_results.json'

# 결과를 저장할 리스트
results = []

# 25번 반복 실행
for i in range(1, count+1):
    best_mse = float('inf')
    best_params = {}

    def model_test():
        global best_mse, best_params  # 함수 내에서 수정 가능하도록 global 선언

        # Loop through hyperparameters
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for al in l2_alpha:
                    print(f'[{i}] Training with batch_size={batch_size}, lr={lr}, al={al}')
                    
                    # Clear previous models
                    tf.keras.backend.clear_session()

                    # Define model with Dropout and Regularization
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],), kernel_regularizer=l2(al)),  # L2 Regularization
                        tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
                        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(al)),
                        tf.keras.layers.Dropout(0.4),
                        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(al)),
                        tf.keras.layers.Dropout(0.25),
                        tf.keras.layers.Dense(1, activation='linear')  # Regression output
                    ])

                    # Compile model with correct learning rate
                    optimizer = Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                    # EarlyStopping to avoid overfitting
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    # Train model
                    history = model.fit(x_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[early_stopping])

                    # Predict and evaluate
                    y_pred = model.predict(x_test).flatten()
                    mse = mean_squared_error(y_test, y_pred)

                    if mse < best_mse:
                        best_mse = mse
                        best_params = {'batch_size': batch_size, 'learning_rate': lr, 'l2_alpha': al}

        # Save the best result from this iteration
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