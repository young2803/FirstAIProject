import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf

df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')
df = df.drop('ReadingTime', axis=1)
df = df[(df['SleepTime'] <= 10) & (df['SleepTime'] >= 3)]

feature = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
label = ['SleepTime']

scaler = MinMaxScaler()
df[feature] = scaler.fit_transform(df[feature])

x_train, x_test, y_train, y_test = train_test_split(df[feature], df[label], test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))