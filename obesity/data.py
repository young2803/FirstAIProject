import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# CSV 파일 불러오기
df = pd.read_csv(r"C:\Users\admin\Desktop\ai data\FirstAIProject\data\Obesity_prediction.csv")

# ✅ 1. 중복 데이터 제거
df.drop_duplicates(inplace=True)

# ✅ 2. 데이터 타입 변환
df["Age"] = df["Age"].astype(int)
df["Height"] = df["Height"].astype(float)
df["Weight"] = df["Weight"].astype(float)

# ✅ 3. 라벨 인코딩 (Yes/No → 0/1 변환)
binary_cols = ["family_history", "FAVC", "SMOKE", "SCC"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

# ✅ 4. 원-핫 인코딩 (범주형 변수 → 더미 변수 생성)
one_hot_cols = ["CAEC", "CALC", "MTRANS"]
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# ✅ 5. 타겟 변수 인코딩 (Obesity 변환)
label_encoder = LabelEncoder()
df["Obesity"] = label_encoder.fit_transform(df["Obesity"])

# ✅ 6. 이상치 처리 (IQR 방법 → 클리핑 적용)
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df[numeric_cols] = df[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

# ✅ 7. 데이터 분리 (훈련 데이터 & 테스트 데이터)
X = df.drop(columns=["Obesity"])
y = df["Obesity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ 8. 훈련 데이터를 다시 훈련 & 검증 데이터로 분리
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# ✅ 9. 데이터 스케일링 (StandardScaler 적용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ✅ 10. 클래스 불균형 처리 (SMOTE 적용)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ✅ 11. 상관관계 분석 후 높은 상관관계 피처 제거
corr_matrix = pd.DataFrame(X_train, columns=X.columns).corr()
high_corr_features = set()
threshold = 0.9

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_features.add(corr_matrix.columns[i])

# ✅ 컬럼명이 정수로 인식되는 문제 해결
drop_columns = [X.columns[i] if isinstance(i, int) else i for i in high_corr_features]
X_train = pd.DataFrame(X_train, columns=X.columns).drop(columns=drop_columns, errors='ignore')
X_val = pd.DataFrame(X_val, columns=X.columns).drop(columns=drop_columns, errors='ignore')
X_test = pd.DataFrame(X_test, columns=X.columns).drop(columns=drop_columns, errors='ignore')

# ✅ 12. 최종 데이터 크기 확인
print("훈련 데이터 크기:", X_train.shape)
print("검증 데이터 크기:", X_val.shape)
print("테스트 데이터 크기:", X_test.shape)
print("타겟 데이터 크기:", y_train.shape, y_val.shape, y_test.shape)

# ✅ 13. 데이터 저장 (CSV 파일)
X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
