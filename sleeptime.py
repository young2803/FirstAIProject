import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')


if 'ReadingTime' in df.columns:
    df = df.drop('ReadingTime', axis=1)

# 수면 시간 이상치 제거 (3시간 이하, 10시간 이상 제거)
df = df[(df['SleepTime'] >= 3) & (df['SleepTime'] <= 10)]

# 정규화할 피처 선택
feature_columns = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']

# 정규화 (Min-Max Scaling)
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# 데이터 분할 (훈련 80%, 테스트 20%)
X = df[feature_columns]
y = df['SleepTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결과 출력 (정규화된 값 확인)
print(df.head())
