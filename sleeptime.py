import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')
df = df.drop('ReadingTime', axis=1)
df = df[(df['SleepTime'] <= 10) & (df['SleepTime'] >= 3)]

feature = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
label = ['SleepTime']

print(df.info())
sns.boxplot(y=df['SleepTime'])
plt.show()