import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Cleaned Training Data
train_data = pd.read_excel('New_data_02_train_cleaned.xlsx', header=None)
X_train = train_data.iloc[:, 0:6].values
y_train = train_data.iloc[:, 6:8].values # Cols 6 & 7 (R and T)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Build and Train Model
model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(16, activation='relu'),
      tf.keras.layers.Dense(2) # 2 outputs: R and T
])

model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)

# 2. Load Cleaned Testing Data (10 lines)
test_data = pd.read_excel('New_data_02_test_cleaned.xlsx', header=None)
X_test = test_data.iloc[:, 0:6].values
y_test_actual = test_data.iloc[:, 6:8].values

# Predict and Evaluate
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

mse_R = mean_squared_error(y_test_actual[:, 0], y_pred[:, 0])
mse_T = mean_squared_error(y_test_actual[:, 1], y_pred[:, 1])

print(f"--- DNN TEST ON 10 UNSEEN LINES ---")
print(f"Average MSE for Reflectance: {mse_R:.6f}")
print(f"Average MSE for Transmittance: {mse_T:.6f}")
