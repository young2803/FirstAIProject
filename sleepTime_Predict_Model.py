import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

def predict_sleep_time(model, scaler, features):
    print("\n사용자의 생활 패턴을 입력하세요:")
    user_input = []
    for feature in features:
        value = float(input(f'{feature} 입력: '))
        user_input.append(value)

    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)

    predicted_sleep_time = model.predict(user_input)[0][0]
    print(f'예측된 수면 시간: {predicted_sleep_time:.2f} 시간')

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop('ReadingTime', axis=1)
    df = df[(df['SleepTime'] <= 10) & (df['SleepTime'] >= 3)]

    features = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
    label = ['SleepTime']

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features, label, scaler

def build_model(input_dim, learning_rate=0.001, l2_alpha=0.0001):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
        Dropout(0.25),
        Dense(1, activation='linear')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_model(model, x_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], verbose=1)
    
    # 모델 학습 후 저장
    model.save('sleeptime_model_1.keras')  # 모델을 파일로 저장
    return model

def load_trained_model():
    return load_model('sleeptime_model_1.keras')  # 저장된 모델 불러오기

if __name__ == "__main__":
    try:
        # 이미 학습된 모델이 있으면 불러오기
        model = load_trained_model()
    except:
        # 학습된 모델이 없으면 새로 학습
        file_path = 'datasets/sleeptime_prediction_dataset.csv'
        df, features, label, scaler = load_and_preprocess_data(file_path)

        x_train = df[features].values
        y_train = df[label].values

        model = build_model(input_dim=x_train.shape[1])
        model = train_model(model, x_train, y_train)

    # 예측 수행
    predict_sleep_time(model, scaler, features)