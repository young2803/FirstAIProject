import pandas as pd
from sklearn.preprocessing import LabelEncoder


# CSV 파일 불러오기
df = pd.read_csv('FirstAIProject/data/Obesity_prediction.csv')

# 데이터 확인
print(df.head())


#1. 데이터 로드 및 기본 정보 확인

# 데이터 크기 확인
print("데이터 크기 (행, 열):", df.shape)

# 컬럼별 데이터 타입 확인
print("\n데이터 타입 정보:")
print(df.info())

# 결측치 확인
print("\n결측치 개수:")
print(df.isnull().sum())

# 중복 데이터 확인
print("\n중복된 행 개수:", df.duplicated().sum())

# 중복 데이터 제거
df.drop_duplicates(inplace=True)

# 데이터 타입 변환
# 'Height', 'Weight', 'Age'가 문자열로 저장된 경우 float로 변환
df["Height"] = df["Height"].astype(float)
df["Weight"] = df["Weight"].astype(float)
df["Age"] = df["Age"].astype(int)

# 데이터 확인
print(df.info())  # 변환된 데이터 타입 확인
print("\n결측치 개수:")
print(df.isnull().sum())  # 결측치가 잘 처리됐는지 확인
print("\n중복된 행 개수:", df.duplicated().sum())  # 중복 데이터가 제거됐는지 확인