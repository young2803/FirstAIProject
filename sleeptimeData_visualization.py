import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression

df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')

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

# 데이터 정보 출력
print(df.info())

# 상관 행렬 계산
corr_matrix = df.corr()
corr_with_label = corr_matrix[label].sort_values(by='SleepTime', ascending=False)
print(corr_with_label)

# 상관 행렬 시각화
plt.figure(figsize=(6, 4))
plt.title('Correlation with Sleep Time')
sns.heatmap(corr_with_label, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# 이상치 시각화 (박스 플롯)
plt.figure(figsize=(12, 6))

for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.xlabel('')

plt.tight_layout()
plt.show()

# 각 Feature들의 중요도(SleepTime에 얼마나 영향을 미치는가?) 계산
f_scores, _ = f_regression(df[features], df[label])
feature_importance = pd.Series(f_scores, index=features).sort_values(ascending=False)
print(feature_importance)