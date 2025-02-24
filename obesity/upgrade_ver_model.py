import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd

# 데이터 불러오기 (전처리 담당)
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv')

# MLP 모델 설계
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(y_train['Obesity'].unique()), activation='softmax')  # 다중분류
])

# 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 결과 출력
model.summary()
test_loss, test_acc = model.evaluate(X_test, y_test)  # 테스트 데이터를 사용하여 모델 평가
print("Test Accuracy:", test_acc)  # 테스트 정확도 출력

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

# 개선된 MLP 모델 설계
model = Sequential({
    Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(len(y_train['Obesity'].unique()), activation='softmax')
})

# Early Stopping 적용
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# 컴파일 & 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# 학습 곡선 확인
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
