import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop

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
x_train, x_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.3, random_state=42)

# Hyperparameters
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.01, 0.1]
optimizers = ['adam', 'rmsprop']

best_mse = float('inf')
best_params = {}

# Loop through hyperparameters
for batch_size in batch_sizes:
    for lr in learning_rates:
        for op in optimizers:
            print(f'Training with batch_size={batch_size}, lr={lr}, optimizer={op}')
            
            # Clear previous models
            tf.keras.backend.clear_session()

            # Define model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')  # Regression output
            ])

            # Select optimizer
            if op == 'adam':
                opt = Adam(learning_rate=lr)
            elif op == 'rmsprop':
                opt = RMSprop(learning_rate=lr)

            # Compile model
            model.compile(optimizer=opt, loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(x_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2, verbose=0)

            # Predict and evaluate
            y_pred = model.predict(x_test).flatten()
            mse = mean_squared_error(y_test, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_params = {'batch_size': batch_size, 'learning_rate': lr, 'optimizer': op}

# Output best parameters
print(f'Best MSE: {best_mse}')
print(f'Best Parameters: {best_params}')