import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/sleeptime_prediction_dataset.csv')
df = df.drop('ReadingTime', axis=1)

feature = ['WorkoutTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
label = ['SleepTime']

corr_matrix = df.corr()

plt.title('Correlation Matrix')
sns.heatmap(corr_matrix, annot=True)
plt.show()