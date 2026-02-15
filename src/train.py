import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import models, layers, callbacks

raw_obesity = pd.read_csv('data/Obesity_prediction.csv')

y = raw_obesity['Obesity']
X = raw_obesity.drop(columns=['Obesity'])

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)

joblib.dump(le, 'models/label_encoder.pkl')


# One-hot encoding
onehot_cols = [
    'Gender', 'family_history', 'FAVC', 
    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]

X = pd.get_dummies(X, columns = onehot_cols, drop_first=True, dtype=int)

train_columns = X.columns
joblib.dump(list(train_columns), "models/train_columns.pkl")


# Train/Test split


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y # 클래스 비율 유지
)


# Scaling

num_cols = [
    'Age',
    'Height',
    'Weight',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE'
]


std_scaler = StandardScaler()
X_train[num_cols] = std_scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = std_scaler.transform(X_test[num_cols])

joblib.dump(std_scaler, 'models/std_scaler.pkl')


# Model definition
class_num = len(set(y_train))

model = models.Sequential()
model.add(keras.Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(class_num, activation='softmax'))

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)



# Training

model_path = 'models/obesity.keras'
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10,
    restore_best_weights=True
    )
checkpoint = callbacks.ModelCheckpoint(
    filepath=model_path, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True
    ) 
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose = 1
)


# Evaluation


test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Loss :", test_loss)
print("Test Acc  :", test_acc)


# classification report
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# plot

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# Accuracy plot
plt.figure()
plt.plot(epochs, acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("figures/accuracy.png", dpi=150)


# Loss plot
plt.figure()
plt.plot(epochs, loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig("figures/loss.png", dpi=150)

print("\nTraining complete.")